from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
from ..services.CareerPredictionService import career_service
from langchain_groq import ChatGroq
import asyncio
import hmac
import json
import os

router = APIRouter(prefix="/career-prediction")


class CareerCandidate(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    match_score: int = Field(ge=0, le=100)
    signals: Dict[str, int] = Field(default_factory=dict)
    fit_reasons: List[str] = Field(default_factory=list, max_length=4)


class CareerAiReviewRequest(BaseModel):
    riasec_code: str = Field(min_length=1, max_length=6)
    strengths: List[str] = Field(default_factory=list, max_length=8)
    candidates: List[CareerCandidate] = Field(min_length=2, max_length=5)


class RankedCareer(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    reasoning: str = Field(min_length=10, max_length=600)


class CareerAiReviewOutput(BaseModel):
    summary: str = Field(min_length=20, max_length=1200)
    ranked_candidates: List[RankedCareer] = Field(min_length=2, max_length=5)
    agreements: List[str] = Field(default_factory=list, max_length=5)
    cautions: List[str] = Field(default_factory=list, max_length=5)
    validation_questions: List[str] = Field(default_factory=list, max_length=5)
    next_actions: List[str] = Field(default_factory=list, max_length=5)


@router.post("/review")
async def review_career_result(payload: CareerAiReviewRequest, x_internal_token: str | None = Header(default=None)):
    """Review an already-scored anonymous result without adding candidates."""
    expected_token = os.getenv("AGENTIC_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(status_code=503, detail="AI review is not configured")
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid internal token")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    candidates = [item.model_dump() for item in payload.candidates]
    allowed_codes = {item["code"] for item in candidates}
    prompt = f"""Bạn là chuyên gia phản biện kết quả định hướng học tập MajorGuide.
Thuật toán đã chấm điểm trước. Hãy đánh giá lại độc lập nhưng CHỈ xếp hạng candidate được cung cấp.
Không thêm ngành mới, không gọi match_score là độ chính xác, không khẳng định chắc chắn tương lai.
Phân tích điểm đồng thuận, điểm có thể gây nhiễu, câu hỏi cần tự kiểm chứng và hành động thực tế.
Không yêu cầu hoặc suy đoán danh tính, sức khỏe, tài chính hay địa chỉ.

Mã RIASEC: {payload.riasec_code}
Tín hiệu nổi bật: {json.dumps(payload.strengths, ensure_ascii=False)}
Ứng viên đã ẩn danh: {json.dumps(candidates, ensure_ascii=False)}
"""
    try:
        model = ChatGroq(api_key=api_key, model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0.15, timeout=25, max_retries=1, max_tokens=1000).with_structured_output(CareerAiReviewOutput)
        review = await asyncio.to_thread(model.invoke, prompt)
        dumped = review.model_dump() if hasattr(review, "model_dump") else dict(review)
        seen, safe_ranking = set(), []
        for item in dumped.get("ranked_candidates", []):
            code = item.get("code")
            if code in allowed_codes and code not in seen:
                seen.add(code)
                safe_ranking.append(item)
        if len(safe_ranking) < 2:
            raise ValueError("AI returned an invalid candidate ranking")
        dumped["ranked_candidates"] = safe_ranking
        return {"success": True, "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), "review": dumped}
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"AI review failed: {type(error).__name__}")


class CareerPredictionRequest(BaseModel):
    answers: Dict[str, int] = Field(
        ...,
        description="Dictionary of question IDs and their answers (1-5 scale). Expected keys: q1 to q40",
        example={
            "q1": 4, "q2": 5, "q3": 4, "q4": 3, "q5": 4,
            "q6": 4, "q7": 5, "q8": 4, "q9": 4, "q10": 3,
            "q11": 4, "q12": 5, "q13": 4, "q14": 5, "q15": 4,
            "q16": 4, "q17": 5, "q18": 3, "q19": 4, "q20": 4,
            "q21": 2, "q22": 3, "q23": 2, "q24": 3, "q25": 4,
            "q26": 5, "q27": 4, "q28": 4, "q29": 3, "q30": 4,
            "q31": 4, "q32": 5, "q33": 4, "q34": 4, "q35": 4,
            "q36": 3, "q37": 4, "q38": 5, "q39": 4, "q40": 4
        }
    )


@router.post("/predict")
async def predict_career(payload: CareerPredictionRequest):
    """
    Dự đoán nhóm ngành nghề phù hợp dựa trên 40 câu trả lời questionnaire.
    
    - **answers**: Dictionary với key là question ID (q1-q40) và value là điểm trả lời (1-5)
    
    Returns:
        - prediction: Nhóm ngành được dự đoán chính
        - top_3: Top 3 nhóm ngành có khả năng cao nhất
        - all_probabilities: Xác suất của tất cả các nhóm ngành
    """
    result = career_service.predict(payload.answers)
    return result


@router.get("/majors")
async def get_available_majors():
    """
    Lấy danh sách tất cả các nhóm ngành có thể dự đoán.
    """
    from ..services.CareerPredictionService import MAJOR_LABELS
    return {
        "majors": [
            {"label": label, "display_name": name}
            for label, name in MAJOR_LABELS.items()
        ],
        "total": len(MAJOR_LABELS)
    }


@router.get("/questions")
async def get_questionnaire():
    """
    Lấy danh sách 40 câu hỏi questionnaire để frontend hiển thị.
    """
    import json
    from pathlib import Path
    
    # Try to read from testaiagent/questions.json if available
    questions_path = Path(__file__).parent.parent.parent / "questions.json"
    
    if questions_path.exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    else:
        # Return basic structure if questions.json not available
        questions = [
            {
                "question_id": f"q{i}",
                "group": f"Group {i // 10 + 1}",
                "text": f"Câu hỏi {i}",
                "traits": []
            }
            for i in range(1, 41)
        ]
    
    return {
        "questions": questions,
        "total": len(questions),
        "scale": {
            "min": 1,
            "max": 5,
            "labels": {
                "1": "Hoàn toàn không đồng ý",
                "2": "Không đồng ý",
                "3": "Bình thường",
                "4": "Đồng ý",
                "5": "Hoàn toàn đồng ý"
            }
        }
    }
