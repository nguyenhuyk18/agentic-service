import joblib
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Label mapping
MAJOR_LABELS = {
    "software_it": "Công nghệ thông tin - Phần mềm",
    "ai_data": "Trí tuệ nhân tạo & Khoa học dữ liệu",
    "cybersecurity_network": "An toàn thông tin & Mạng",
    "engineering_mechanical": "Kỹ thuật cơ khí",
    "engineering_electrical": "Kỹ thuật điện - Điện tử",
    "construction_architecture": "Xây dựng & Kiến trúc",
    "business_management": "Quản trị kinh doanh",
    "marketing_media": "Marketing & Truyền thông",
    "finance_accounting": "Tài chính - Kế toán",
    "logistics_supplychain": "Logistics & Chuỗi cung ứng",
    "design_multimedia": "Thiết kế & Đồ họa đa phương tiện",
    "content_media": "Sáng tạo nội dung & Media",
    "language_international": "Ngôn ngữ & Quan hệ quốc tế",
    "psychology_education": "Tâm lý học & Giáo dục",
    "law_publicservice": "Luật & Dịch vụ công",
    "healthcare_biology": "Y tế & Sinh học"
}


class CareerPredictionService:
    _instance = None
    _model = None
    _feature_columns = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        # Path: apps/services/CareerPredictionService.py -> root: agentic-ai/
        base_path = Path(__file__).resolve().parent.parent.parent
        model_path = base_path / "career_model.pkl"
        feature_path = base_path / "feature_columns.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {feature_path}")

        self._model = joblib.load(model_path)
        self._feature_columns = joblib.load(feature_path)
        print(f"[CareerPredictionService] Model loaded successfully")
        print(f"[CareerPredictionService] Feature columns: {len(self._feature_columns)} features")

    def _validate_answers(self, answers: Dict[str, int]) -> tuple[bool, str]:
        required_questions = [f"q{i}" for i in range(1, 41)]

        for q in required_questions:
            if q not in answers:
                return False, f"Thiếu câu trả lời cho {q}"

        for q in required_questions:
            val = answers[q]
            if not isinstance(val, (int, float)) or val < 1 or val > 5:
                return False, f"Câu trả lời cho {q} phải là số từ 1 đến 5"

        return True, "OK"

    def predict(self, answers: Dict[str, int]) -> Dict[str, Any]:
        valid, msg = self._validate_answers(answers)
        if not valid:
            return {
                "success": False,
                "error": msg
            }

        input_data = {q: answers[q] for q in self._feature_columns}

        df = np.array([list(input_data.values())])

        prediction = self._model.predict(df)
        prediction_proba = self._model.predict_proba(df)

        predicted_label = str(prediction[0][0]) if hasattr(prediction[0], '__iter__') else str(prediction[0])
        class_labels = self._model.classes_

        proba_dict = {}
        for i, label in enumerate(class_labels):
            proba_dict[label] = round(float(prediction_proba[0][i]), 4)

        sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_proba[:3]

        return {
            "success": True,
            "prediction": {
                "primary": {
                    "label": predicted_label,
                    "display_name": MAJOR_LABELS.get(predicted_label, predicted_label),
                    "confidence": round(float(max(prediction_proba[0])), 4)
                },
                "top_3": [
                    {
                        "label": label,
                        "display_name": MAJOR_LABELS.get(label, label),
                        "confidence": round(conf, 4)
                    }
                    for label, conf in top3
                ],
                "all_probabilities": proba_dict
            }
        }


career_service = CareerPredictionService()
