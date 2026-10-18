"""Prediction test cases DD-01..DD-12: chosen answers -> predicted career family.

Each case lists the families Holland theory considers consistent with the profile;
the model's top family must be one of them. Flat profiles (every answer the same)
carry no information, so no family may look confident.
"""
import json
import unittest
from pathlib import Path

from ml.career.inference import CareerFamilyPredictor

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "tests" / "fixtures" / "prediction_scenarios.json").read_text(encoding="utf-8"))


class PredictionScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CareerFamilyPredictor.load(ROOT / "ml" / "artifacts")

    def test_cases(self):
        for case in CASES:
            with self.subTest(case=case["id"], name=case["name"]):
                p = self.predictor.proba(self.predictor.means(case["answers"]))[0]
                top = self.predictor.classes[int(p.argmax())]
                if case["expectTop1"]:
                    self.assertIn(top, case["expectTop1"])
                else:
                    self.assertLess(float(p.max()), 0.2, "a flat profile must not produce a confident prediction")


if __name__ == "__main__":
    unittest.main()
