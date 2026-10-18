"""Inference shared by offline evaluation and the FastAPI service.

proba():      features -> model -> temperature scaling -> label-shift correction
explain():    occlusion attribution per RIASEC scale for the top families
next_scale(): adaptive testing; keeps RIASEC coverage balanced and uses the
              expected entropy reduction of the family distribution as tie-break.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from .features import SCALES, scale_features

NO_ANSWER_BUCKET = 9  # buckets 0..8 = running mean 1.0, 1.5, ..., 5.0


def mean_bucket(values: list[int]) -> int:
    if not values:
        return NO_ANSWER_BUCKET
    return int(np.clip(round(np.mean(values) * 2) - 2, 0, 8))


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=-1)


class CareerFamilyPredictor:
    def __init__(self, model, classes, temperature=1.0, source_prior=None, beta=0.0, next_answer=None, card=None):
        self.model = model
        self.classes = list(classes)
        self.temperature = float(temperature)
        self.source_prior = np.asarray(source_prior if source_prior is not None else np.ones(len(classes)) / len(classes))
        self.beta = float(beta)
        self.next_answer = np.asarray(next_answer) if next_answer is not None else None  # (6, 10, 5)
        self.card = card or {}

    @classmethod
    def load(cls, directory: Path) -> "CareerFamilyPredictor":
        directory = Path(directory)
        card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
        cat = json.loads((directory / "cat_tables.json").read_text(encoding="utf-8"))
        return cls(
            joblib.load(directory / "career_family_model.joblib"),
            card["classes"], card["temperature"], card["source_prior"], card["label_shift_beta"],
            cat["next_answer"], card,
        )

    # ------------------------------------------------------------------ scoring
    def proba(self, means: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(scale_features(means))
        z = np.log(np.clip(raw, 1e-12, 1)) / self.temperature
        # Label-shift correction: the training sample over-represents some majors
        # (it was collected on a psychology website); divide out its prior.
        z = z - self.beta * np.log(self.source_prior)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    @staticmethod
    def means(answers: dict[str, list[int]]) -> np.ndarray:
        return np.array([[((np.mean(answers[s]) - 1) / 4) if answers.get(s) else np.nan for s in SCALES]])

    def explain(self, answers: dict[str, list[int]], top: int = 3) -> list[dict]:
        """For each top family: how much each answered scale pushes it up/down
        (probability change when the scale is replaced by the person's average)."""
        base_means = self.means(answers)
        base = self.proba(base_means)[0]
        answered = [j for j, s in enumerate(SCALES) if answers.get(s)]
        elevation = np.nanmean(base_means)
        variants = np.repeat(base_means, len(answered), axis=0)
        for row, j in enumerate(answered):
            variants[row, j] = elevation
        occluded = self.proba(variants) if answered else np.empty((0, len(self.classes)))
        result = []
        for c in np.argsort(-base)[:top]:
            contributions = {SCALES[j]: round(float(base[c] - occluded[row, c]), 4) for row, j in enumerate(answered)}
            result.append({"family": self.classes[c], "probability": round(float(base[c]), 4), "scale_contributions": contributions})
        return result

    # ---------------------------------------------------------------- adaptive
    def next_scale(self, answers: dict[str, list[int]], allowed: list[str] | None = None, balanced: bool = True) -> list[dict]:
        """Rank scales by expected information gain (EIG) about the family.

        balanced=True (deployed): scales with the fewest answers come first and EIG
        only breaks ties. Pure EIG keeps confirming the current leader — it makes the
        model more confident, not more accurate (see reports/adaptive_testing.png)."""
        if self.next_answer is None:
            raise RuntimeError("next_answer tables are not loaded")
        allowed = [s for s in (allowed or SCALES) if s in SCALES]
        base_means = self.means(answers)
        current = float(entropy(self.proba(base_means))[0])
        rows, weights, owners = [], [], []
        for scale in allowed:
            j = SCALES.index(scale)
            history = list(answers.get(scale, []))
            p_v = self.next_answer[j, mean_bucket(history)]
            for v in range(1, 6):
                means = base_means.copy()
                means[0, j] = (np.mean(history + [v]) - 1) / 4
                rows.append(means[0])
                weights.append(p_v[v - 1])
                owners.append(scale)
        if not rows:
            return []
        after = entropy(self.proba(np.array(rows)))
        ranking = []
        for scale in allowed:
            idx = [i for i, owner in enumerate(owners) if owner == scale]
            expected = float(np.dot(np.array(weights)[idx], after[idx]) / np.sum(np.array(weights)[idx]))
            ranking.append({"scale": scale, "answered": len(answers.get(scale, [])), "expected_information_gain": round(current - expected, 5)})
        if balanced:
            return sorted(ranking, key=lambda item: (item["answered"], -item["expected_information_gain"]))
        return sorted(ranking, key=lambda item: -item["expected_information_gain"])


def next_answer_tables(item_values: dict[str, np.ndarray], rng: np.random.Generator, alpha=1.0) -> np.ndarray:
    """P(next item = v | bucket of the running mean of previously answered items
    of the same scale), estimated with random answer orders. Shape (6, 10, 5)."""
    tables = np.full((6, 10, 5), alpha)
    for j, scale in enumerate(SCALES):
        values = item_values[scale]  # (n, 8)
        n = len(values)
        for _ in range(4):
            order = np.argsort(rng.random(values.shape), axis=1)
            shuffled = np.take_along_axis(values, order, axis=1)
            k = rng.integers(0, 8, size=n)  # number of earlier answers
            for prefix in range(8):
                rows = k == prefix
                if not rows.any():
                    continue
                if prefix == 0:
                    buckets = np.full(rows.sum(), NO_ANSWER_BUCKET)
                else:
                    buckets = np.clip(np.round(shuffled[rows, :prefix].mean(axis=1) * 2) - 2, 0, 8).astype(int)
                np.add.at(tables[j], (buckets, shuffled[rows, prefix] - 1), 1)
    return tables / tables.sum(axis=2, keepdims=True)
