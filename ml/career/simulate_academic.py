"""Semi-synthetic simulation: does adding grades improve career-family prediction?

There is no public dataset linking Vietnamese students' grades to their majors, so
this is a SIMULATION built from two real sources and one explicit assumption:

  * real interest profiles + real majors : RIASEC 2018 (62,312 labelled people)
  * real grades                          : 2024 THPT national exam (~1.06M candidates)
  * assumption (parameter λ)             : a person studying family f is drawn from the
    candidates who can apply with one of f's admission combinations, with probability
    ∝ exp(λ·z), z = probit of their best national percentile on those combinations.
    λ = 0: only eligibility (having sat one of f's combinations) links grades to the
    major; λ = 1: students of f also score about 1 SD higher on those combinations.
    "independent": grades are drawn from all candidates, unrelated to the major — the
    worst case, showing what the fusion costs if the assumption is entirely wrong.

For every λ the test split (same 80/20 split as train.py) is scored by
  1. interest only           (deployed MLP, 5 items per scale as in the app)
  2. knowledge-based fusion  (deployed rule: P_interest · exp(γ·(fit − 0.5)))
  3. trained joint model     (gradient boosting on interest + grades; an upper bound
                              that is only valid under the simulated assumption)

γ is chosen once on a validation split for the best average over three scenarios
(unrelated grades, λ = 0.5, λ = 1) and then kept fixed for every row. Results are a simulation, not an accuracy
claim about real Vietnamese students.

Usage:  python -m career.simulate_academic [--quick]      (after prepare_data, train, prepare_scores)
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .academic import ABS_WEIGHT, COMBINATIONS, CORE_WEIGHT, FAMILY_ACADEMICS, SUBJECTS, academic_fit, export_reference, fuse
from .features import SCALES, scale_features
from .inference import CareerFamilyPredictor
from .taxonomy import FAMILIES

ROOT = Path(__file__).resolve().parent.parent
DATA, ARTIFACTS, REPORTS = ROOT / "data", ROOT / "artifacts", ROOT / "reports"
TS_REFERENCE = ROOT.parent.parent / "apps/user-access/src/app/modules/career-assessment/academic-reference.ts"
TS_PARITY = ROOT.parent.parent / "apps/user-access/src/app/modules/career-assessment/academic-parity.json"
SEED = 42
CLASSES = list(FAMILIES)
ITEMS = {s: [f"{s}{i}" for i in range(1, 9)] for s in SCALES}
# "independent": grades drawn from all candidates, unrelated to the major (worst case).
LAMBDAS = ["independent", 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
SUBJECT_COLS = list(SUBJECTS)


def topk(p, y, k=3):
    return float((np.argsort(-p, axis=1)[:, :k] == y[:, None]).any(axis=1).mean())


def macro_topk(p, y, k=3):
    return float(np.mean([topk(p[y == c], y[y == c], k) for c in np.unique(y)]))


def harmonic(p, y):
    a, b = topk(p, y), macro_topk(p, y)
    return 2 * a * b / (a + b)


# ------------------------------------------------------------------ grades side
def pool_tables(pool: pd.DataFrame, quantiles: dict) -> dict:
    """Vectorised academic fit of every 2024 candidate for every family (exam source)."""
    n = len(pool)
    totals = {c: pool[list(s)].sum(axis=1, skipna=False).to_numpy() for c, s in COMBINATIONS.items()}
    pct = {c: np.interp(np.nan_to_num(t, nan=-1), quantiles[c], np.linspace(0, 1, 101)) for c, t in totals.items()}
    own_mean = pool[SUBJECT_COLS].mean(axis=1).to_numpy()
    w_abs = ABS_WEIGHT["thi_thu"]
    fit = np.zeros((n, len(CLASSES)), dtype=np.float32)
    best_pct = np.full((n, len(CLASSES)), np.nan, dtype=np.float32)
    for j, family in enumerate(CLASSES):
        combos = FAMILY_ACADEMICS[family]["combos"]
        stack = np.column_stack([np.nan_to_num(totals[c], nan=-np.inf) for c in combos])
        idx = stack.argmax(axis=1)
        best_total = stack[np.arange(n), idx]
        ok = np.isfinite(best_total)
        best_p = np.column_stack([pct[c] for c in combos])[np.arange(n), idx]
        relative = np.clip(0.5 + (best_total / 3 - own_mean) / 4, 0, 1)
        combo_fit = w_abs * best_p + (1 - w_abs) * relative
        core = FAMILY_ACADEMICS[family]["core"]
        if core:
            core_mean = pool[core].mean(axis=1).to_numpy()  # mean of the core subjects present
            core_strength = np.clip(0.5 + (core_mean - own_mean) / 4, 0, 1)
            combo_fit = np.where(np.isnan(core_mean), combo_fit, (1 - CORE_WEIGHT) * combo_fit + CORE_WEIGHT * core_strength)
        fit[:, j] = np.where(ok, combo_fit, 0)
        # Percentile across the family's own combinations, used by the selection assumption.
        best_pct[:, j] = np.where(ok, np.column_stack([np.where(np.isnan(totals[c]), -1, pct[c]) for c in combos]).max(axis=1), np.nan)
    return {"fit": fit, "best_pct": best_pct}


def sample_grades(y: np.ndarray, best_pct: np.ndarray, lam, rng: np.random.Generator) -> np.ndarray:
    """Index of a 2024 candidate for each person, given their family and λ."""
    if lam == "independent":
        return rng.integers(0, len(best_pct), size=len(y))
    out = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        members = np.where(y == c)[0]
        u = best_pct[:, c]
        eligible = ~np.isnan(u)
        z = norm.ppf(np.clip(u[eligible], 0.005, 0.995))
        w = np.exp(lam * z)
        candidates = np.where(eligible)[0]
        out[members] = rng.choice(candidates, size=len(members), p=w / w.sum())
    return out


def interest_means(df: pd.DataFrame, rng: np.random.Generator, k=5) -> np.ndarray:
    """Scale means (0..1) from k random items per scale (the app asks 5)."""
    out = np.empty((len(df), 6))
    for j, s in enumerate(SCALES):
        values = df[ITEMS[s]].to_numpy(dtype=float)
        order = np.argsort(rng.random(values.shape), axis=1)[:, :k]
        out[:, j] = (np.take_along_axis(values, order, axis=1).mean(axis=1) - 1) / 4
    return out


# ------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.time()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATA / "riasec_clean.csv")
    y_all = df["family"].map({f: i for i, f in enumerate(CLASSES)}).to_numpy()
    train_df, test_df, y_train, y_test = train_test_split(df, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    if args.quick:
        train_df, y_train = train_df.iloc[:12000], y_train[:12000]
        test_df, y_test = test_df.iloc[:4000], y_test[:4000]

    predictor = CareerFamilyPredictor.load(ARTIFACTS)
    means_train, means_test = interest_means(train_df, rng), interest_means(test_df, rng)
    p_int_train, p_int_test = predictor.proba(means_train), predictor.proba(means_test)
    feat_train, feat_test = scale_features(means_train), scale_features(means_test)

    quantiles = json.loads((ARTIFACTS / "combo_quantiles.json").read_text(encoding="utf-8"))
    pool = pd.read_csv(DATA / "scores_2024.csv.gz", dtype="float32")
    tables = pool_tables(pool, quantiles)
    aptitude = np.array([bool(FAMILY_ACADEMICS[f].get("aptitude")) for f in CLASSES])
    subjects = pool[SUBJECT_COLS].to_numpy()

    # Choose γ on a validation part of train. Instead of optimising for one assumed λ,
    # take the γ with the best AVERAGE over three scenarios — grades unrelated to the
    # major, a weak link (λ = 0.5) and a moderate link (λ = 1) — so the deployed rule
    # stays useful if the assumption is wrong.
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=SEED)
    scenarios = ["independent", 0.5, 1.0]
    val_fits = {lam: tables["fit"][sample_grades(y_train[val_idx], tables["best_pct"], lam, rng)] for lam in scenarios}
    gamma_table = {
        g: {str(lam): round(harmonic(fuse(p_int_train[val_idx], val_fits[lam], g, aptitude), y_train[val_idx]), 4) for lam in scenarios}
        for g in GAMMAS
    }
    gamma = max(GAMMAS, key=lambda g: np.mean(list(gamma_table[g].values())))
    gamma_scores = {g: round(float(np.mean(list(v.values()))), 4) for g, v in gamma_table.items()}
    print(f"gamma={gamma}")
    for g, v in gamma_table.items():
        print(f"  γ={g}: {v} mean={gamma_scores[g]}")

    rows = []
    prior = np.bincount(y_train, minlength=len(CLASSES)) / len(y_train)
    for lam in LAMBDAS:
        g_train = sample_grades(y_train, tables["best_pct"], lam, rng)
        g_test = sample_grades(y_test, tables["best_pct"], lam, rng)
        fit_test = tables["fit"][g_test]
        joint = HistGradientBoostingClassifier(max_iter=150 if args.quick else 300, learning_rate=0.08, random_state=SEED)
        joint.fit(np.hstack([feat_train, tables["fit"][g_train], subjects[g_train]]), y_train)
        p_joint = joint.predict_proba(np.hstack([feat_test, fit_test, subjects[g_test]]))
        # Same label-shift correction as the deployed interest model.
        p_joint = p_joint / prior ** predictor.beta
        p_joint /= p_joint.sum(axis=1, keepdims=True)
        methods = {
            "interest_only": p_int_test,
            "knowledge_fusion": fuse(p_int_test, fit_test, gamma, aptitude),
            "trained_joint_model": p_joint,
        }
        row = {"lambda": lam}
        for name, p in methods.items():
            row[name] = {"top3": round(topk(p, y_test), 4), "macro_top3": round(macro_topk(p, y_test), 4), "top1": round(topk(p, y_test, 1), 4)}
        rows.append(row)
        print(f"λ={lam}: " + "  ".join(f"{k}={v['top3']:.3f}/{v['macro_top3']:.3f}" for k, v in row.items() if k != "lambda"))

    report = {
        "type": "SIMULATION — not an accuracy claim about Vietnamese students",
        "sources": {
            "interests_and_majors": "Open-Source Psychometrics Project RIASEC 2018 (real)",
            "grades": "THPT national exam 2024, ~1.06M candidates (real distributions and subject correlations)",
            "link_between_them": "assumption with parameter λ (see module docstring)",
        },
        "items_per_scale": 5,
        "gamma_selected": gamma,
        "gamma_selection": "max mean harmonic(top3, macro_top3) over scenarios independent, λ=0.5, λ=1 (validation split)",
        "gamma_grid_by_scenario": {str(g): v for g, v in gamma_table.items()},
        "results": rows,
        "duration_sec": round(time.time() - started, 1),
    }
    (REPORTS / "academic_simulation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(rows)

    meta = {"gamma_selection": "robust mean over independent, λ=0.5, λ=1", "exam_year": 2024, "candidates": int(len(pool))}
    export_reference(TS_REFERENCE, quantiles, gamma, meta)
    # Cross-language parity vectors: the TypeScript port must reproduce these fits.
    cases = [
        ({"toan": 9, "vat_li": 8.5, "hoa_hoc": 8, "ngu_van": 6.5, "ngoai_ngu": 7}, "thi_thu"),
        ({"toan": 6, "ngu_van": 8.5, "lich_su": 8, "dia_li": 8.75, "ngoai_ngu": 6.5}, "thi_thu"),
        ({"toan": 9, "hoa_hoc": 9, "sinh_hoc": 9.25, "ngu_van": 5.5, "ngoai_ngu": 6}, "thi_thu"),
        ({"toan": 8.5, "ngu_van": 8.8, "ngoai_ngu": 9.2, "vat_li": 8, "hoa_hoc": 8.3, "sinh_hoc": 8.1, "lich_su": 8.6, "dia_li": 8.9, "gdcd": 9.4}, "hoc_ba"),
    ]
    TS_PARITY.write_text(json.dumps([{"scores": s, "source": src, "expected": academic_fit(s, quantiles, src)} for s, src in cases], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {report['duration_sec']}s; γ={gamma}; wrote {TS_REFERENCE.name}")


def plot(rows):
    rows = [r for r in rows if r["lambda"] != "independent"]
    lams = [r["lambda"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = {"interest_only": "Chỉ sở thích (hiện tại)", "knowledge_fusion": "Sở thích + học lực (tri thức tuyển sinh, triển khai)", "trained_joint_model": "Mô hình học chung (cận trên, chỉ đúng trong mô phỏng)"}
    for ax, metric, title in ((axes[0], "top3", "Top-3 accuracy"), (axes[1], "macro_top3", "Top-3 trung bình theo nhóm ngành")):
        for name, label in labels.items():
            ax.plot(lams, [r[name][metric] for r in rows], marker="o", label=label)
        ax.set_xlabel("λ — mức ảnh hưởng giả định của học lực tới việc chọn ngành")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("MÔ PHỎNG: sở thích thật (RIASEC) + điểm thật (THPT 2024), liên kết theo giả định λ", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "academic_simulation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
