"""Academic profile: subject scores -> how well a student's grades fit each career family.

Admission in Vietnam is decided per "tổ hợp xét tuyển" (3-subject combination,
total out of 30). Each family lists the combinations most commonly used by
universities for its majors and its core subjects. The academic fit of a family is

    combo = w_abs * percentile(best combination total vs. 2024 national exam)
          + (1 - w_abs) * relative strength(best combination vs. the student's own average)
    fit   = (1 - CORE_WEIGHT) * combo + CORE_WEIGHT * core strength
            (core subjects of the family vs. the student's own average)

Two families sharing the same best combination (e.g. medicine and psychology both
accept B00) are separated by their core subjects (biology/chemistry vs. literature).

`percentile` uses real 2024 THPT exam results (see prepare_scores.py). School
report grades ("học bạ") are inflated compared with exam scores, so they rely
mostly on relative strength (which subjects stand out for this student).

This module is the single source of truth; export_reference() generates the
TypeScript copy used by the NestJS service.
"""
import json
from pathlib import Path

import numpy as np

SUBJECTS = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Tiếng Anh",
    "vat_li": "Vật lí",
    "hoa_hoc": "Hóa học",
    "sinh_hoc": "Sinh học",
    "lich_su": "Lịch sử",
    "dia_li": "Địa lí",
    "gdcd": "GDCD / GDKT&PL",
}

COMBINATIONS = {
    "A00": ("toan", "vat_li", "hoa_hoc"),
    "A01": ("toan", "vat_li", "ngoai_ngu"),
    "B00": ("toan", "hoa_hoc", "sinh_hoc"),
    "B08": ("toan", "sinh_hoc", "ngoai_ngu"),
    "C00": ("ngu_van", "lich_su", "dia_li"),
    "D01": ("ngu_van", "toan", "ngoai_ngu"),
    "D07": ("toan", "hoa_hoc", "ngoai_ngu"),
    "D14": ("ngu_van", "lich_su", "ngoai_ngu"),
    "D15": ("ngu_van", "dia_li", "ngoai_ngu"),
}

# Common admission combinations per family (đề án tuyển sinh of public universities)
# and core subjects. `aptitude`: admission usually also needs an aptitude exam
# (drawing, performing arts, sport) that grades cannot capture.
FAMILY_ACADEMICS = {
    "it_software": {"combos": ["A00", "A01", "D07", "D01"], "core": ["toan"]},
    "engineering": {"combos": ["A00", "A01", "D07"], "core": ["toan", "vat_li"]},
    "architecture_construction": {"combos": ["A00", "A01", "D01"], "core": ["toan"], "aptitude": True},
    "business_management": {"combos": ["A00", "A01", "D01", "D07"], "core": ["toan", "ngoai_ngu"]},
    "marketing_communication": {"combos": ["A01", "D01", "C00", "D14"], "core": ["ngu_van", "ngoai_ngu"]},
    "finance_economics": {"combos": ["A00", "A01", "D01", "D07"], "core": ["toan"]},
    "arts_design": {"combos": ["D01", "A01", "C00"], "core": ["ngu_van"], "aptitude": True},
    "humanities_language": {"combos": ["D01", "D14", "D15", "C00"], "core": ["ngoai_ngu", "ngu_van"]},
    "psychology_counseling": {"combos": ["C00", "D01", "B00", "B08"], "core": ["ngu_van"]},
    "education": {"combos": ["A00", "B00", "C00", "D01"], "core": []},
    "social_work": {"combos": ["C00", "D01", "D14", "D15"], "core": ["ngu_van"]},
    "law_public": {"combos": ["C00", "A00", "A01", "D01"], "core": ["ngu_van"]},
    "health_medicine": {"combos": ["B00", "B08", "A00"], "core": ["sinh_hoc", "hoa_hoc"]},
    "life_sciences": {"combos": ["B00", "A00", "D07", "B08"], "core": ["sinh_hoc", "hoa_hoc"]},
    "physical_sciences_math": {"combos": ["A00", "A01", "B00", "D07"], "core": ["toan"]},
    "hospitality_tourism": {"combos": ["D01", "C00", "A01", "D15"], "core": ["ngoai_ngu"]},
    "sports_fitness": {"combos": ["B00", "D01", "C00"], "core": [], "aptitude": True},
}

# Weight of the national percentile vs. relative strength, per score source.
ABS_WEIGHT = {"thi_thu": 0.6, "hoc_ba": 0.2}
# Share of the fit given to the family's core subjects (when the student entered them).
CORE_WEIGHT = 0.25


def combo_totals(scores: dict[str, float | None]) -> dict[str, float]:
    """Totals (0..30) of every combination whose 3 subjects are all present."""
    totals = {}
    for code, subjects in COMBINATIONS.items():
        values = [scores.get(s) for s in subjects]
        if all(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in values):
            totals[code] = round(float(sum(values)), 2)
    return totals


def percentile(code: str, total: float, quantiles: dict[str, list[float]]) -> float:
    """Share (0..1) of 2024 candidates with that combination scoring below `total`."""
    q = np.asarray(quantiles[code])  # 101 values: 0th..100th percentile
    return float(np.clip(np.interp(total, q, np.linspace(0, 1, len(q))), 0, 1))


def academic_fit(scores: dict[str, float | None], quantiles: dict[str, list[float]], source: str = "thi_thu") -> dict[str, dict]:
    """Per family: fit (0..1), best combination, its total and national percentile."""
    totals = combo_totals(scores)
    present = [v for v in scores.values() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not totals or not present:
        return {}
    own_mean = float(np.mean(present))
    w_abs = ABS_WEIGHT.get(source, ABS_WEIGHT["thi_thu"])
    result = {}
    for family, info in FAMILY_ACADEMICS.items():
        available = [c for c in info["combos"] if c in totals]
        if not available:
            result[family] = {"fit": 0.0, "combo": None, "total": None, "percentile": None}
            continue
        best = max(available, key=lambda c: totals[c])
        pct = percentile(best, totals[best], quantiles)
        relative = float(np.clip(0.5 + (totals[best] / 3 - own_mean) / 4, 0, 1))
        fit = w_abs * pct + (1 - w_abs) * relative
        core = [scores[s] for s in info["core"] if scores.get(s) is not None and not (isinstance(scores[s], float) and np.isnan(scores[s]))]
        if core:
            core_strength = float(np.clip(0.5 + (np.mean(core) - own_mean) / 4, 0, 1))
            fit = (1 - CORE_WEIGHT) * fit + CORE_WEIGHT * core_strength
        result[family] = {
            "fit": round(fit, 4),
            "combo": best,
            "total": totals[best],
            "percentile": round(pct, 4),
        }
    return result


def fuse(interest_proba: np.ndarray, fit: np.ndarray, gamma: float, aptitude_mask: np.ndarray | None = None) -> np.ndarray:
    """Log-linear fusion: p(f) ∝ P_interest(f) · exp(γ · (fit(f) − 0.5)).
    Families that also require an aptitude exam use half the academic weight."""
    g = np.full(fit.shape[-1], gamma) if aptitude_mask is None else np.where(aptitude_mask, gamma / 2, gamma)
    z = np.log(np.clip(interest_proba, 1e-12, 1)) + g * (fit - 0.5)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def export_reference(path: Path, quantiles: dict, gamma: float, meta: dict):
    """Write the TypeScript module consumed by apps/user-access (generated file)."""
    payload = {
        "subjects": SUBJECTS,
        "combinations": {k: list(v) for k, v in COMBINATIONS.items()},
        "families": FAMILY_ACADEMICS,
        "absWeight": ABS_WEIGHT,
        "coreWeight": CORE_WEIGHT,
        "gamma": gamma,
        "quantiles": quantiles,
        "meta": meta,
    }
    path.write_text(
        "// GENERATED by agentic-ai/ml/career/simulate_academic.py — do not edit by hand.\n"
        "// Combination percentiles come from the 2024 THPT national exam results.\n"
        f"export const ACADEMIC_REFERENCE = {json.dumps(payload, ensure_ascii=False, indent=1)} as const;\n",
        encoding="utf-8",
    )
