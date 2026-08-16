from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict
from ..services.CareerPredictionService import career_service

router = APIRouter(prefix="/career-prediction")


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
