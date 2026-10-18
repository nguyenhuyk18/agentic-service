"""Refresh the frozen ML probabilities of the prediction test cases (TC-DD-xx).

apps/user-access/.../prediction-scenarios.json holds, per scenario, the answers, optional
grades, the model's family probabilities ("mlFamilies") and the expected results.
After retraining, run this to re-score every scenario with the new model, then run the
TypeScript suite: any change in the top 3 shows up as a failing case to review.

Usage:  python -m career.export_scenarios        (run from agentic-ai/ml)
"""
import json
from pathlib import Path

from .inference import CareerFamilyPredictor
from .taxonomy import FAMILIES

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT.parent.parent / "apps/user-access/src/app/modules/career-assessment/prediction-scenarios.json"


def main():
    predictor = CareerFamilyPredictor.load(ROOT / "artifacts")
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in cases:
        p = predictor.proba(predictor.means(case["answers"]))[0]
        ranked = sorted(zip(predictor.classes, p.tolist()), key=lambda kv: -kv[1])
        case["mlFamilies"] = [{"code": c, "name": FAMILIES[c], "probability": round(v, 4)} for c, v in ranked]
    FIXTURE.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    for case in cases:
        print(case["id"], case["mlFamilies"][0]["code"], case["mlFamilies"][0]["probability"])


if __name__ == "__main__":
    main()
