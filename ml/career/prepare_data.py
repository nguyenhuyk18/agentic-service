"""Download, clean and label the open RIASEC dataset.

Source: Open-Source Psychometrics Project, RIASEC_data12Dec2018 (CC BY-NC-SA 4.0),
https://openpsychometrics.org/_rawdata/ — 145k online responses to 48 Holland
interest items (1=Dislike .. 5=Enjoy) with a free-text "major" field.

Usage:  python -m career.prepare_data           (run from agentic-ai/ml)
Output: data/riasec_clean.csv + reports/data_report.json
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .taxonomy import FAMILIES, map_major

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
URL = "https://openpsychometrics.org/_rawdata/RIASEC_data12Dec2018.zip"
SCALES = list("RIASEC")
ITEMS = [f"{s}{i}" for s in SCALES for i in range(1, 9)]


def load_raw() -> pd.DataFrame:
    raw = DATA / "RIASEC_data12Dec2018.csv"
    if not raw.exists():
        print(f"Downloading {URL}")
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            raw.write_bytes(archive.read("RIASEC_data12Dec2018/data.csv"))
    return pd.read_csv(raw, sep="\t", low_memory=False)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    steps = {"raw": len(df)}
    items = df[ITEMS]
    df = df[items.isin([1, 2, 3, 4, 5]).all(axis=1)]
    steps["complete_items"] = len(df)
    # VCL6, VCL9, VCL12 are fake words: claiming to know them signals careless answering.
    df = df[(df[["VCL6", "VCL9", "VCL12"]] == 0).all(axis=1)]
    steps["passed_fake_word_check"] = len(df)
    df = df[df[ITEMS].std(axis=1) > 0]  # straight-lining
    steps["not_straight_lining"] = len(df)
    df = df[df["testelapse"].between(48, 3600)]  # >= 1s per item, < 1h
    steps["plausible_duration"] = len(df)
    df = df[df["age"].between(13, 80)]
    steps["plausible_age"] = len(df)
    df = df.assign(family=df["major"].map(map_major))
    labelled = df[df["family"].notna()].copy()
    steps["labelled_major"] = len(labelled)
    unmapped = df.loc[df["family"].isna(), "major"].dropna().astype(str).str.strip().str.lower()
    return labelled, steps, unmapped.value_counts().head(40).to_dict()


def main():
    DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df, steps, unmapped_examples = clean(load_raw())
    columns = ITEMS + ["education", "age", "gender", "testelapse", "major", "family"]
    df[columns].to_csv(DATA / "riasec_clean.csv", index=False)

    counts = df["family"].value_counts()
    report = {
        "source": URL,
        "license": "CC BY-NC-SA 4.0 (Open-Source Psychometrics Project)",
        "cleaning_steps": steps,
        "families": {f: {"name": FAMILIES[f], "n": int(counts.get(f, 0))} for f in FAMILIES},
        "imbalance_ratio": round(float(counts.max() / counts.min()), 1),
        "scale_means": {s: round(float(df[[f"{s}{i}" for i in range(1, 9)]].values.mean()), 3) for s in SCALES},
        # Most frequent majors left out of training, for auditing the mapping rules.
        "unmapped_examples": unmapped_examples,
    }
    (REPORTS / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(steps, indent=2))
    print(counts.to_string())


if __name__ == "__main__":
    main()
