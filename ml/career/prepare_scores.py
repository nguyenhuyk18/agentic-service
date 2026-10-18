"""Prepare the 2024 THPT national exam scores (public results of the Ministry of
Education and Training, ~1.06M candidates) for the academic simulation.

Input : data/Diemthi2024_processed.csv — per-candidate subject scores, as re-published at
        https://huggingface.co/spaces/ngocminhta/National-Grad-Score-Ranking-1
Output: data/scores_2024.csv.gz   — the 9 subject columns only (candidate number,
                                    province and region are dropped)
        artifacts/combo_quantiles.json — 0..100th percentiles of every combination
        reports/scores_report.json

Usage:  python -m career.prepare_scores        (run from agentic-ai/ml)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .academic import COMBINATIONS, SUBJECTS

ROOT = Path(__file__).resolve().parent.parent
DATA, ARTIFACTS, REPORTS = ROOT / "data", ROOT / "artifacts", ROOT / "reports"
SOURCE = "https://huggingface.co/spaces/ngocminhta/National-Grad-Score-Ranking-1"


def main():
    raw = DATA / "Diemthi2024_processed.csv"
    if not raw.exists():
        raise SystemExit(f"Missing {raw}. Download Diemthi2024_processed.csv from {SOURCE}")
    df = pd.read_csv(raw, usecols=list(SUBJECTS), dtype="float32")
    df = df[df.notna().sum(axis=1) >= 3]
    df = df[((df >= 0) & (df <= 10)) | df.isna()].dropna(how="all")
    df.to_csv(DATA / "scores_2024.csv.gz", index=False, compression="gzip")

    quantiles, report = {}, {"source": SOURCE, "candidates": int(len(df)), "subjects": {}, "combinations": {}}
    for subject in SUBJECTS:
        column = df[subject].dropna()
        report["subjects"][subject] = {"n": int(len(column)), "mean": round(float(column.mean()), 3), "std": round(float(column.std()), 3)}
    for code, subjects in COMBINATIONS.items():
        totals = df[list(subjects)].dropna().sum(axis=1)
        quantiles[code] = np.round(np.quantile(totals, np.linspace(0, 1, 101)), 3).tolist()
        report["combinations"][code] = {"n": int(len(totals)), "median": round(float(totals.median()), 2), "p90": round(float(totals.quantile(0.9)), 2)}
    taken = df[["vat_li", "lich_su"]].notna()
    report["block_share"] = {"khtn": round(float(taken["vat_li"].mean()), 4), "khxh": round(float(taken["lich_su"].mean()), 4)}
    report["subject_correlation"] = df.corr().round(3).to_dict()

    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    (ARTIFACTS / "combo_quantiles.json").write_text(json.dumps(quantiles), encoding="utf-8")
    (REPORTS / "scores_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidates", "block_share")}, indent=2))
    print(json.dumps(report["combinations"], indent=2))


if __name__ == "__main__":
    main()
