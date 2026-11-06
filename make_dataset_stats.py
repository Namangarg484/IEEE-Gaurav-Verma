"""
make_dataset_stats.py
=====================
Cohort facts for the paper, computed from the data on disk (never typed):
subjects, sex, age range, records, windows, glucose range, diabetic subjects,
total and held-out CGM readings.  Writes results/dataset_stats.json, which
make_results_macros.py turns into \\mendNSubjects, \\bigNRecordsAll, etc.

    python make_dataset_stats.py
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from ppg_data import load_bigideas, load_mendeley, load_stuba

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


def basic(ws, meta_sex="gender"):
    m = ws.meta.drop_duplicates("subject")
    d = {"n_subjects": int(ws.meta.subject.nunique()), "n_records": int(len(ws.meta)), "n_windows": int(len(ws)),
         "glucose_min": float(ws.meta.glucose.min()), "glucose_max": float(ws.meta.glucose.max()),
         "glucose_mean": float(ws.meta.glucose.mean()), "glucose_std": float(ws.meta.glucose.std(ddof=0)),
         "hypo_n": int((ws.meta.glucose < 70).sum()), "hyper_n": int((ws.meta.glucose > 180).sum()),
         "high140_n": int((ws.meta.glucose > 140).sum())}
    if meta_sex in m:
        d["n_male"] = int((m[meta_sex] == "Male").sum()); d["n_female"] = int((m[meta_sex] == "Female").sum())
    if "age" in m:
        d["age_min"], d["age_max"] = float(m.age.min()), float(m.age.max())
    if "diabetes" in m:
        d["n_diabetic"] = int((m.diabetes.astype(str).str.lower() == "yes").sum())
    rec_per = ws.meta.groupby("subject").size()
    d["subjects_le2_records"] = int((rec_per <= 2).sum())
    return d


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    out = {"mendeley": basic(load_mendeley()), "stuba": basic(load_stuba())}
    big = load_bigideas(max_windows_per_egv=2)
    out["bigideas"] = basic(big)
    out["bigideas"]["n_readings_all"] = int(len(big.meta))
    zip_path = os.path.join(ROOT, "data.nosync", "bigideas", "bigideas-1.1.3.zip")
    if os.path.exists(zip_path):
        import io, zipfile
        import pandas as pd
        z = zipfile.ZipFile(zip_path)
        demo = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("Demographics.csv")][0])))
        out["bigideas"]["n_female"] = int((demo.Gender.str.upper() == "FEMALE").sum())
        out["bigideas"]["n_male"] = int((demo.Gender.str.upper() == "MALE").sum())
        out["bigideas"]["hba1c_min"], out["bigideas"]["hba1c_max"] = float(demo.HbA1c.min()), float(demo.HbA1c.max())
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "dataset_stats.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
