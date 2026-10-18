"""Train, compare and export the career-family model.

Usage:  python -m career.train [--quick]      (run from agentic-ai/ml, after prepare_data)

Pipeline
  1. Psychometrics: Cronbach's alpha per scale + 6-factor EFA (varimax).
  2. Split 80/20 stratified; simulate adaptive tests by averaging 2-8 random items/scale.
  3. Compare: prior, current rule-based cosine (production baseline), multinomial
     logistic regression, random forest, MLP, CatBoost — 5-fold CV on train.
  4. Fit the best model, temperature-scale probabilities on a validation split,
     evaluate on the untouched test set (full answers and 2-item answers),
     fairness by gender, per-family recall, confusion matrix.
  5. Adaptive testing: categorical Naive Bayes tables + simulation of expected
     information gain selection vs. fixed order vs. random order.
  6. Export artifacts/ (model, model card, CAT tables) and reports/.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, SCALES, scale_features
from .inference import CareerFamilyPredictor, next_answer_tables
from .taxonomy import FAMILIES, FAMILY_OF_PROFILE, families_of

ROOT = Path(__file__).resolve().parent.parent
DATA, ARTIFACTS, REPORTS = ROOT / "data", ROOT / "artifacts", ROOT / "reports"
SEED = 42
CLASSES = list(FAMILIES)
ITEMS = {s: [f"{s}{i}" for i in range(1, 9)] for s in SCALES}

# The app's career profiles (code + RIASEC prototype) live in TypeScript; they are
# parsed so the "current system" baseline is exactly what production uses.
APP_PROFILES_TS = ROOT.parent.parent / "apps/user-access/src/app/modules/career-assessment/career-assessment.data.ts"


def load_app_profiles(path: Path = APP_PROFILES_TS) -> dict[str, list[float]]:
    import re
    text = path.read_text(encoding="utf-8")
    found = re.findall(r"p\(\s*'([a-z_]+)',\s*'[^']*',\s*\[([^\]]+)\]", text)
    profiles = {code: [float(v) for v in vector.split(",")] for code, vector in found}
    if len(profiles) != len(FAMILY_OF_PROFILE):
        raise ValueError(f"Expected {len(FAMILY_OF_PROFILE)} profiles in {path}, found {len(profiles)}")
    return profiles


# --------------------------------------------------------------------------- data
def sample_means(df: pd.DataFrame, rng: np.random.Generator, k_min=2, k_max=8) -> np.ndarray:
    """Scale means (0..1) from k random items per scale, k ~ U[k_min, k_max]."""
    out = np.empty((len(df), 6))
    for j, scale in enumerate(SCALES):
        values = df[ITEMS[scale]].to_numpy(dtype=float)
        k = rng.integers(k_min, k_max + 1, size=len(df))
        keys = rng.random(values.shape)
        ranks = keys.argsort(axis=1).argsort(axis=1)
        mask = ranks < k[:, None]
        out[:, j] = ((values * mask).sum(axis=1) / mask.sum(axis=1) - 1) / 4
    return out


def full_means(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([(df[ITEMS[s]].mean(axis=1) - 1) / 4 for s in SCALES])


# ------------------------------------------------------------------- psychometrics
def cronbach_alpha(items: np.ndarray) -> float:
    k = items.shape[1]
    return float(k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1)))


def psychometrics(df: pd.DataFrame) -> dict:
    alphas = {s: round(cronbach_alpha(df[ITEMS[s]].to_numpy(float)), 3) for s in SCALES}
    all_items = [i for s in SCALES for i in ITEMS[s]]
    sample = df[all_items].sample(min(20000, len(df)), random_state=SEED)
    fa = FactorAnalysis(n_components=6, rotation="varimax", random_state=SEED)
    fa.fit(StandardScaler().fit_transform(sample))
    loadings = pd.DataFrame(fa.components_.T, index=all_items)
    # Each factor is labelled by the scale whose items load on it most.
    factor_scale = {f: max(SCALES, key=lambda s: loadings.loc[ITEMS[s], f].abs().mean()) for f in loadings.columns}
    correct = sum(factor_scale[loadings.loc[item].abs().idxmax()] == item[0] for item in all_items)
    return {
        "cronbach_alpha": alphas,
        "efa_items_loading_on_own_scale": f"{correct}/{len(all_items)}",
        "efa_factor_labels": list(factor_scale.values()),
    }


# -------------------------------------------------------------------------- metrics
def topk(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-proba, axis=1)[:, :k] == y[:, None]).any(axis=1).mean())


def ece(proba: np.ndarray, y: np.ndarray, bins=15) -> float:
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(total)


def macro_topk(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    """Top-k recall averaged over families, so small families count as much as psychology."""
    return float(np.mean([topk(proba[y == c], y[y == c], k) for c in np.unique(y)]))


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "top1": round(topk(proba, y, 1), 4),
        "top3": round(topk(proba, y, 3), 4),
        "top5": round(topk(proba, y, 5), 4),
        "macro_top3": round(macro_topk(proba, y, 3), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "balanced_acc": round(float(recall_score(y, pred, average="macro")), 4),
        "log_loss": round(float(log_loss(y, np.clip(proba, 1e-9, 1), labels=range(len(CLASSES)))), 4),
        "ece": round(ece(proba, y), 4),
    }


# --------------------------------------------------------------------------- models
class PriorModel:
    def fit(self, X, y, **_):
        self.p = np.bincount(y, minlength=len(CLASSES)) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


class RuleBaseline:
    """The current production interest scorer: cosine(user RIASEC, profile
    prototype) for all 40 app profiles; a family scores its best profile."""

    def fit(self, X, y, **_):
        profiles = load_app_profiles()
        self.protos = np.array([profiles[p] for p in profiles])
        self.member = np.zeros((len(profiles), len(CLASSES)), dtype=bool)
        for i, p in enumerate(profiles):
            for family in families_of(p):
                self.member[i, CLASSES.index(family)] = True
        return self

    def predict_proba(self, X):
        user = X[:, :6]
        sim = (user @ self.protos.T) / (np.linalg.norm(user, axis=1, keepdims=True) * np.linalg.norm(self.protos, axis=1) + 1e-9)
        family_sim = np.where(self.member[None], sim[:, :, None], -1).max(axis=1)
        z = np.exp((family_sim - family_sim.max(axis=1, keepdims=True)) * 30)
        return z / z.sum(axis=1, keepdims=True)


def make_models(quick: bool):
    it = 300 if quick else 800
    return {
        "prior (đoán theo tần suất)": lambda: PriorModel(),
        "rule-based cosine (hệ thống hiện tại)": lambda: RuleBaseline(),
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random forest": lambda: RandomForestClassifier(n_estimators=150 if quick else 300, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED)),
        "catboost": lambda: CatBoostClassifier(iterations=it, depth=6, learning_rate=0.08, loss_function="MultiClass", random_seed=SEED, verbose=0, thread_count=-1),
    }


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(proba, 1e-12, 1))

    def nll(t):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

    return float(minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x)


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, 1)) / t
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


# ------------------------------------------------------------- adaptive testing (CAT)
def simulate_cat(predictor: CareerFamilyPredictor, df: pd.DataFrame, y: np.ndarray, n=1500, core=2, extra=12) -> dict:
    """Every strategy starts like the app: `core` items per scale (12 core items),
    then asks `extra` more interest items chosen by the strategy."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    values = {s: df[ITEMS[s]].to_numpy(int)[idx] for s in SCALES}
    hits = {k: np.zeros(extra + 1) for k in ("balanced_information_gain", "information_gain", "round_robin", "random")}
    for row in range(len(idx)):
        for strategy in hits:
            answers = {s: [int(v) for v in values[s][row, :core]] for s in SCALES}
            for step in range(extra + 1):
                proba = predictor.proba(predictor.means(answers))[0]
                hits[strategy][step] += y[idx[row]] in np.argsort(-proba)[:3]
                if step == extra:
                    break
                allowed = [s for s in SCALES if len(answers[s]) < 8]
                if strategy == "balanced_information_gain":  # deployed
                    scale = predictor.next_scale(answers, allowed, balanced=True)[0]["scale"]
                elif strategy == "information_gain":
                    scale = predictor.next_scale(answers, allowed, balanced=False)[0]["scale"]
                elif strategy == "round_robin":
                    scale = allowed[step % len(allowed)]
                else:
                    scale = allowed[rng.integers(len(allowed))]
                answers[scale].append(int(values[scale][row, len(answers[scale])]))
    return {
        "questions": [6 * core + step for step in range(extra + 1)],
        **{k: (v / len(idx)).round(4).tolist() for k, v in hits.items()},
    }


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer iterations, for a smoke test")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    started = time.time()

    df = pd.read_csv(DATA / "riasec_clean.csv")
    y_all = df["family"].map({f: i for i, f in enumerate(CLASSES)}).to_numpy()
    report = {"n_samples": len(df), "classes": CLASSES}

    print("1/6 psychometrics")
    report["psychometrics"] = psychometrics(df)

    print("2/6 split + adaptive simulation")
    train_df, test_df, y_train, y_test = train_test_split(df, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    rng = np.random.default_rng(SEED)
    # Two simulated adaptive sessions per training person, so models learn noisy short-test scores.
    X_train = np.vstack([scale_features(sample_means(train_df, rng)) for _ in range(2)])
    y_train_aug = np.concatenate([y_train, y_train])
    X_test_full = scale_features(full_means(test_df))
    X_test_short = scale_features(sample_means(test_df, np.random.default_rng(SEED + 1), 2, 2))
    X_test_adaptive = scale_features(sample_means(test_df, np.random.default_rng(SEED + 2), 3, 4))

    print("3/6 5-fold cross-validation")
    models = make_models(args.quick)
    cv = StratifiedKFold(n_splits=3 if args.quick else 5, shuffle=True, random_state=SEED)
    groups = np.concatenate([np.arange(len(y_train))] * 2)  # keep both copies of a person in the same fold
    cv_results = {}
    for name, factory in models.items():
        scores = []
        for fold_train, fold_val in cv.split(np.zeros(len(y_train)), y_train):
            tr = np.isin(groups, fold_train)
            va = np.isin(groups, fold_val) & (np.arange(len(groups)) < len(y_train))
            model = factory().fit(X_train[tr], y_train_aug[tr])
            scores.append(evaluate(model.predict_proba(X_train[va]), y_train_aug[va]))
        cv_results[name] = {m: [round(float(np.mean([s[m] for s in scores])), 4), round(float(np.std([s[m] for s in scores])), 4)] for m in scores[0]}
        print(f"   {name:42s} top3={cv_results[name]['top3'][0]:.3f} macroF1={cv_results[name]['macro_f1'][0]:.3f} logloss={cv_results[name]['log_loss'][0]:.3f}")
    report["cross_validation"] = cv_results
    learnable = [n for n in models if not n.startswith(("prior", "rule"))]
    best = min(learnable, key=lambda n: cv_results[n]["log_loss"][0])
    report["selected_model"] = best
    print(f"   selected: {best}")

    print("4/6 final fit + calibration + test")
    fit_idx, cal_idx = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=SEED)
    fit_mask = np.isin(groups, fit_idx)
    cal_mask = np.isin(groups, cal_idx) & (np.arange(len(groups)) < len(y_train))
    final = models[best]().fit(X_train[fit_mask], y_train_aug[fit_mask])
    temperature = fit_temperature(final.predict_proba(X_train[cal_mask]), y_train_aug[cal_mask])
    source_prior = np.bincount(y_train, minlength=len(CLASSES)) / len(y_train)
    # Choose the label-shift strength on the calibration split: best balance (harmonic
    # mean) between overall top-3 and family-averaged top-3.
    cal_means = X_train[cal_mask][:, :6]
    beta_scores = {}
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        candidate = CareerFamilyPredictor(final, CLASSES, temperature, source_prior, beta)
        p = candidate.proba(cal_means)
        a, b = topk(p, y_train_aug[cal_mask], 3), macro_topk(p, y_train_aug[cal_mask], 3)
        beta_scores[beta] = round(2 * a * b / (a + b), 4)
    beta = max(beta_scores, key=beta_scores.get)
    report["label_shift"] = {"beta_grid_harmonic_top3": beta_scores, "selected_beta": beta}
    item_values = {s: train_df[ITEMS[s]].to_numpy(int) for s in SCALES}
    predictor = CareerFamilyPredictor(final, CLASSES, temperature, source_prior, beta, next_answer_tables(item_values, rng))
    baseline = RuleBaseline().fit(X_train, y_train_aug)
    tests = {"full_48_items": X_test_full, "adaptive_3_4_items_per_scale": X_test_adaptive, "short_2_items_per_scale": X_test_short}
    report["test"] = {}
    for label, X in tests.items():
        raw = final.predict_proba(X)
        report["test"][label] = {
            "model_uncalibrated": evaluate(raw, y_test),
            "model_calibrated": evaluate(apply_temperature(raw, temperature), y_test),
            "model_deployed": evaluate(predictor.proba(X[:, :6]), y_test),
            "rule_baseline": evaluate(baseline.predict_proba(X), y_test),
            "prior": evaluate(PriorModel().fit(None, y_train).predict_proba(X), y_test),
        }
    report["temperature"] = round(temperature, 4)
    proba_test = predictor.proba(X_test_adaptive[:, :6])
    report["per_family_top3_recall"] = {
        CLASSES[c]: round(topk(proba_test[y_test == c], y_test[y_test == c], 3), 3) for c in range(len(CLASSES))
    }
    report["fairness_top3_by_gender"] = {
        label: round(topk(proba_test[test_df["gender"].to_numpy() == g], y_test[test_df["gender"].to_numpy() == g], 3), 4)
        for g, label in ((1, "male"), (2, "female"))
    }
    # Model-agnostic importance: drop in top-3 accuracy when one scale is replaced by the person's average.
    base_top3 = topk(proba_test, y_test, 3)
    importance = {}
    for j, scale in enumerate(SCALES):
        occluded = X_test_adaptive[:, :6].copy()
        occluded[:, j] = occluded.mean(axis=1)
        importance[scale] = round(base_top3 - topk(predictor.proba(occluded), y_test, 3), 4)
    report["scale_importance_top3_drop"] = importance

    print("5/6 adaptive testing simulation")
    report["adaptive_testing_top3"] = simulate_cat(predictor, test_df, y_test, n=200 if args.quick else 1500)

    print("6/6 export")
    model_path = ARTIFACTS / "career_family_model.joblib"
    joblib.dump(final, model_path)
    card = {
        "name": "majorguide-career-family",
        "version": time.strftime("%Y.%m.%d"),
        "model": best,
        "features": FEATURES,
        "classes": CLASSES,
        "class_names": FAMILIES,
        "temperature": report["temperature"],
        "source_prior": np.round(source_prior, 6).tolist(),
        "label_shift_beta": beta,
        "profile_families": {p: families_of(p) for p in FAMILY_OF_PROFILE},
        "training_data": "Open-Source Psychometrics Project RIASEC 2018 (CC BY-NC-SA 4.0), cleaned & labelled — see reports/data_report.json",
        "metrics_adaptive_test": report["test"]["adaptive_3_4_items_per_scale"]["model_deployed"],
        "intended_use": "Gợi ý nhóm ngành để khám phá, không phải kết luận. Không dùng để tuyển chọn hay sàng lọc.",
        "limitations": [
            "Dữ liệu là người dùng Internet nói tiếng Anh 2015-2018, chủ yếu ngoài Việt Nam.",
            "Nhãn là ngành đang/đã học, không phải mức độ hài lòng hay thành công.",
            "Chỉ dùng sở thích RIASEC; giá trị nghề và phong cách làm việc do luật trong app xử lý.",
            "Không dùng giới tính, tuổi làm đặc trưng.",
        ],
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "cat_tables.json").write_text(json.dumps({
        "scales": SCALES,
        "description": "P(next answer=v | scale, bucket of running mean); bucket 9 = no answer yet",
        "next_answer": np.round(predictor.next_answer, 6).tolist(),
    }), encoding="utf-8")
    report["duration_sec"] = round(time.time() - started, 1)
    (REPORTS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plots(report, final, X_test_adaptive, y_test, temperature, proba_test)
    print(f"done in {report['duration_sec']}s -> {model_path}")


def plots(report, model, X, y, temperature, proba):
    names = [FAMILIES[c] for c in CLASSES]
    # Confusion matrix (row-normalised)
    cm = confusion_matrix(y, proba.argmax(axis=1), labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Ma trận nhầm lẫn (chuẩn hóa theo hàng) — mô hình triển khai, 3-4 câu/thang")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # Model comparison
    cv = report["cross_validation"]
    labels = list(cv)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(labels))
    for offset, metric in ((-0.2, "top1"), (0.2, "top3")):
        ax.bar(xs + offset, [cv[n][metric][0] for n in labels], 0.38, yerr=[cv[n][metric][1] for n in labels], label=metric)
    ax.set_xticks(xs, labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (5-fold CV)")
    ax.legend()
    ax.set_title("So sánh mô hình")
    fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png", dpi=130)
    plt.close(fig)

    # Adaptive testing curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cat = report["adaptive_testing_top3"]
    for name in ("balanced_information_gain", "information_gain", "round_robin", "random"):
        ax.plot(cat["questions"], cat[name], marker="o", markersize=3, label=name)
    ax.set_xlabel("số câu hỏi sở thích đã trả lời (12 câu lõi + câu thích ứng)")
    ax.set_ylabel("top-3 accuracy")
    ax.set_title("Hỏi thích ứng: chiến lược chọn thang RIASEC tiếp theo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "adaptive_testing.png", dpi=130)
    plt.close(fig)

    # Reliability diagram
    raw = model.predict_proba(X)
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, p in (("chưa hiệu chỉnh", raw), (f"temperature={temperature:.2f}", apply_temperature(raw, temperature))):
        conf, correct = p.max(axis=1), p.argmax(axis=1) == y
        edges = np.linspace(0, 1, 11)
        mids, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() > 20:
                mids.append(conf[mask].mean())
                accs.append(correct[mask].mean())
        ax.plot(mids, accs, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("độ tin cậy dự đoán")
    ax.set_ylabel("tỉ lệ đúng thực tế")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
