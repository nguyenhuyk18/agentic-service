"""Career-family model v2 (trained in ml/career, see ml/README.md).

Called by the user-access service during an adaptive assessment; it never
receives identity data, only interest answers grouped by RIASEC scale.
"""
import hmac
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml.career.features import SCALES
from ml.career.inference import CareerFamilyPredictor
from ml.career.taxonomy import FAMILIES, FAMILY_OF_PROFILE, families_of

router = APIRouter(prefix="/career-prediction/v2")
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"


@lru_cache(maxsize=1)
def get_predictor() -> CareerFamilyPredictor:
    return CareerFamilyPredictor.load(ARTIFACTS)


def require_internal_token(token: str | None):
    expected = os.getenv("AGENTIC_INTERNAL_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Career model is not configured")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid internal token")


def predictor_or_503() -> CareerFamilyPredictor:
    try:
        return get_predictor()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Career model artifacts are missing; run ml/career/train.py")


class InterestAnswers(BaseModel):
    answers: Dict[str, List[int]] = Field(description="RIASEC scale -> answers on a 1..5 scale")

    @field_validator("answers")
    @classmethod
    def valid_answers(cls, value):
        clean = {}
        for scale, items in value.items():
            if scale not in SCALES:
                raise ValueError(f"Unknown RIASEC scale {scale}")
            if len(items) > 8 or any(not isinstance(v, int) or v < 1 or v > 5 for v in items):
                raise ValueError("Each scale accepts at most 8 answers between 1 and 5")
            clean[scale] = items
        if not any(clean.values()):
            raise ValueError("At least one interest answer is required")
        return clean


class NextScaleRequest(InterestAnswers):
    allowed: List[str] = Field(default_factory=lambda: list(SCALES))


@router.post("/score")
async def score(payload: InterestAnswers, x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    predictor = predictor_or_503()
    proba = predictor.proba(predictor.means(payload.answers))[0]
    by_family = dict(zip(predictor.classes, proba.tolist()))
    top = max(by_family.values())
    # Profile score: probability of its family (or families) relative to the best family, 0..1.
    profiles = {p: round(min(1.0, sum(by_family[f] for f in families_of(p)) / top), 4) for p in FAMILY_OF_PROFILE}
    ranked = sorted(by_family.items(), key=lambda kv: -kv[1])
    return {
        "model": {"name": predictor.card.get("name"), "version": predictor.card.get("version"), "type": predictor.card.get("model")},
        "families": [
            {"code": f, "name": FAMILIES[f], "probability": round(p, 4),
             "profiles": [code for code in FAMILY_OF_PROFILE if f in families_of(code)]}
            for f, p in ranked
        ],
        "profiles": profiles,
        "explanations": predictor.explain(payload.answers),
        "answered_items": sum(len(v) for v in payload.answers.values()),
    }


@router.post("/next-scale")
async def next_scale(payload: NextScaleRequest, x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    predictor = predictor_or_503()
    return {"ranking": predictor.next_scale(payload.answers, payload.allowed)}


@router.get("/model-card")
async def model_card():
    card = predictor_or_503().card
    return {key: card.get(key) for key in ("name", "version", "model", "classes", "class_names", "metrics_adaptive_test", "intended_use", "limitations", "training_data")}
