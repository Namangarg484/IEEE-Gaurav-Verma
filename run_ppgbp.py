"""
run_ppgbp.py
============
Generalisation of the leakage audit to a second physiological regression task:
cuff blood pressure from finger PPG on the public PPG-BP database (Liang et
al., Sci. Data 5:180020, 2018; figshare 10.6084/m9.figshare.5459299, CC0).

Why this task: unlike glucose, systolic blood pressure has well-documented
between-subject structure (age, arterial stiffness, pulse morphology), so it
tests whether the null-anchored harness *detects* signal when it exists --
the property a negative glucose result depends on.

Data: 219 subjects, one cuff SBP/DBP reading each, three 2.1-s PPG segments
recorded at 1 kHz (sample-and-hold, so the effective bandwidth is lower).
Each segment is band-passed 0.5-8 Hz, resampled to 64 Hz (134 samples) and
z-scored, exactly like the glucose cohorts but shorter.

Protocols (the ladder for this task)
    segment_random   5-fold over segments: the same subject (and hence the same
                     label) appears in train and test  -> identity leak
    subject_kfold    5-fold grouped by subject (honest)
Every metric is computed per subject (one reference each) next to the
train-mean null, with the same subject-cluster bootstrap as the glucose rows.

Models: mean (null), ridge on hand-crafted PPG features, ridge on age/sex/
height/weight, ridge on both, gradient boosting on both, 1-D CNN.

    python run_ppgbp.py --targets sbp dbp --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from bootstrap_ci import cluster_bootstrap
from ppg_data import FS, WindowSet, _zscore_rows, handcrafted_features, make_views, to_common_rate
from run_subject_cv import OUT_DIR, device, merge_json, metrics, train_fold

log = logging.getLogger("ppgbp")
ROOT = os.path.dirname(os.path.abspath(__file__))
PPGBP_DIR = os.path.join(ROOT, "data.nosync", "ppgbp", "Data File")


def load_ppgbp(target: str) -> tuple[WindowSet, np.ndarray]:
    tab = pd.read_excel(os.path.join(PPGBP_DIR, "PPG-BP dataset.xlsx"), header=1)
    tab.columns = [str(c).strip() for c in tab.columns]
    col = {"sbp": "Systolic Blood Pressure(mmHg)", "dbp": "Diastolic Blood Pressure(mmHg)"}[target]
    Xs, ys, subj, rec, demo, meta = [], [], [], [], [], []
    for _, r in tab.iterrows():
        sid = int(r["subject_ID"])
        files = sorted(glob.glob(os.path.join(PPGBP_DIR, "0_subject", f"{sid}_*.txt")))
        if not files:
            continue
        sex = 1.0 if str(r["Sex(M/F)"]).strip().lower().startswith("m") else 0.0
        d = [float(r["Age(year)"]), float(r["Height(cm)"]), float(r["Weight(kg)"]), sex]
        for f in files:
            raw = np.array([float(v) for v in open(f).read().split() if v.strip()], np.float64)   # files end with a tab
            raw = raw[np.isfinite(raw)]
            if len(raw) < 1000 or raw.std() < 1e-6:
                continue
            Xs.append(to_common_rate(raw, 1000))
            ys.append(float(r[col])); subj.append(str(sid)); rec.append(str(sid)); demo.append(d)
        meta.append({"record": str(sid), "subject": str(sid), "glucose": float(r[col]), "age": d[0],
                     "height": d[1], "weight": d[2], "sex": "Male" if sex else "Female"})
    L = min(len(x) for x in Xs)
    X = _zscore_rows(np.stack([x[:L] for x in Xs]))
    ws = WindowSet(X, np.array(ys, np.float32), np.array(subj), np.array(rec), pd.DataFrame(meta),
                   source=f"PPG-BP ({target.upper()})")
    log.info(ws.describe().replace("glucose", target.upper()))
    return ws, np.array(demo, np.float32)


def folds_for(ws, protocol, seed):
    if protocol == "segment_random":
        return list(KFold(5, shuffle=True, random_state=seed).split(ws.X))
    if protocol == "subject_kfold":
        return list(GroupKFold(5).split(ws.X, ws.y, ws.subject))
    raise ValueError(protocol)


def fit_classical(name, F, y, tr, te):
    if name == "mean":
        return np.full(len(te), y[tr].mean(), np.float32)
    if name.startswith("gbm"):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                          l2_regularization=1.0, random_state=0)
        return m.fit(F[tr], y[tr]).predict(F[te]).astype(np.float32)
    sc = StandardScaler().fit(F[tr])
    return RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(F[tr]), y[tr]).predict(sc.transform(F[te])).astype(np.float32)


def subject_level(pred, ws):
    df = pd.DataFrame({"record": ws.record, "subject": ws.subject, "y": ws.y, "pred": pred})
    return df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"),
                                                                 n_windows=("pred", "size"))


def bp_metrics(df):
    """Regression metrics on a mmHg target (Clarke zones are meaningless here and dropped)."""
    m = metrics(df)
    return {k: v for k, v in m.items() if not k.startswith("zone_") and k != "mard_pct"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["sbp", "dbp"])
    ap.add_argument("--protocols", nargs="+", default=["segment_random", "subject_kfold"])
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "demo", "ridge+demo", "gbm+demo", "1d"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    path = os.path.join(OUT_DIR, "ppgbp_summary.json")
    dev = device()
    for target in args.targets:
        ws, demo = load_ppgbp(target)
        feats = handcrafted_features(ws.X)
        F = {"ridge": feats, "demo": demo, "ridge+demo": np.hstack([feats, demo]), "gbm+demo": np.hstack([feats, demo])}
        views = make_views(ws.X, which=("x1",)) if "1d" in args.models else None
        for protocol in args.protocols:
            null_df = None
            for name in args.models:
                for seed in (args.seeds if name == "1d" else [0]):
                    folds = folds_for(ws, protocol, seed)
                    pred = np.full(len(ws), np.nan, np.float32)
                    t0 = time.time()
                    for tr, te in folds:
                        if name == "1d":
                            pred[te] = train_fold("1d", views, ws.y, tr, te, args.epochs, 32, 1e-3, seed, dev)
                        else:
                            pred[te] = fit_classical(name, F.get(name), ws.y, tr, te)
                    df = subject_level(pred, ws)
                    tag = f"{target}_{protocol}_{name}_seed{seed}"
                    df.to_csv(os.path.join(OUT_DIR, f"ppgbp_{tag}.csv"), index=False)
                    m = bp_metrics(df)
                    if name == "mean":
                        null_df = df
                    elif null_df is not None:
                        b = cluster_bootstrap(df, null_df, np.random.default_rng(seed))
                        m.update(diff_vs_null=b["diff_vs_null"], diff_ci=b["diff_ci"], mae_ci=b["mae_ci"])
                    m.update(target=target, protocol=protocol, model=name, seed=seed, seconds=round(time.time() - t0, 1),
                             n_subjects=int(df.subject.nunique()), n_segments=int(len(ws)))
                    merge_json(path, tag, m)
                    log.info("RESULT %-36s MAE %.2f  r %.3f  delta %s", tag, m["mae"], m.get("pearson_r", np.nan),
                             f"{m['diff_vs_null']:+.2f} [{m['diff_ci'][0]:+.2f},{m['diff_ci'][1]:+.2f}]" if "diff_ci" in m else "--")


if __name__ == "__main__":
    main()
