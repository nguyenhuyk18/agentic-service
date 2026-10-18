"""Feature construction shared by training and serving.

The app asks an adaptive subset of interest items, so a user may answer only
2-8 items per RIASEC scale. Features are therefore scale-level:
mean (0..1) of the answered items of each scale, plus profile shape features
used in Holland's theory (differentiation, consistency of the top-2 code).
"""
import numpy as np

SCALES = list("RIASEC")
# Holland hexagon order R-I-A-S-E-C: adjacent types are "consistent".
_HEXAGON_DISTANCE = {
    (a, b): min(abs(i - j), 6 - abs(i - j))
    for i, a in enumerate(SCALES) for j, b in enumerate(SCALES)
}
FEATURES = (
    [f"{s}_mean" for s in SCALES]
    + [f"{s}_rel" for s in SCALES]
    + ["differentiation", "elevation", "consistency"]
)


def scale_features(means: np.ndarray) -> np.ndarray:
    """means: (n, 6) array of scale means in [0, 1] (NaN = scale not answered)."""
    means = np.asarray(means, dtype=float)
    filled = np.where(np.isnan(means), np.nanmean(means, axis=1, keepdims=True), means)
    filled = np.nan_to_num(filled, nan=0.5)
    elevation = filled.mean(axis=1, keepdims=True)
    relative = filled - elevation  # ipsative profile: what stands out for this person
    differentiation = filled.max(axis=1, keepdims=True) - filled.min(axis=1, keepdims=True)
    order = np.argsort(-filled, axis=1)
    consistency = np.array([
        [1 - _HEXAGON_DISTANCE[(SCALES[row[0]], SCALES[row[1]])] / 3] for row in order
    ])
    return np.hstack([filled, relative, differentiation, elevation, consistency])


def means_from_answers(answers: dict[str, list[int]]) -> np.ndarray:
    """answers: {"R": [1..5, ...], ...} -> (1, 6) scale means in [0, 1]."""
    row = []
    for scale in SCALES:
        values = [v for v in answers.get(scale, []) if v is not None]
        row.append((np.mean(values) - 1) / 4 if values else np.nan)
    return np.array([row])
