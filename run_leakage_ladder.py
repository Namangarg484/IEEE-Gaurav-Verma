"""
run_leakage_ladder.py
=====================
How much does each kind of leakage inflate PPG-glucose metrics?  The same
model is evaluated under protocols that leak progressively less.

Mendeley (67 recordings, 23 subjects)
    window      5-fold over windows: windows of the SAME recording in train and test
    record      5-fold grouped by recording: same SUBJECT in train and test
    subject     leave-one-subject-out (the honest protocol; = run_subject_cv.py)
    + a 'subject-mean (leaky)' predictor that returns the test subject's own
      training mean, i.e. the ceiling a record-level leak lets a model memorise.

BIG IDEAs (16 subjects, CGM every 5 min)
    reading     5-fold over CGM readings (neighbouring readings on both sides)
    chrono0     chronological 70/30 per subject, no guard gap
    chrono      chronological 70/30 with a 1-h guard gap (the honest protocol)

Usage
-----
    python run_leakage_ladder.py --dataset mendeley --models mean 1d fusion --seeds 0 1 2
    python run_leakage_ladder.py --dataset bigideas --models mean 1d --epochs 15

Writes results/leakage_ladder.json (one entry per dataset/protocol/model/seed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from ppg_data import load_bigideas, load_mendeley, make_views
from run_bigideas_cv import chronological_split, subject_mean_baseline
from run_subject_cv import OUT_DIR, device, merge_json, metrics, recording_level, train_fold, _subset

log = logging.getLogger("ladder")
NETS = ("fusion", "1d", "2d", "3d")


def mendeley_folds(ws, protocol, seed):
    if protocol == "window":
        return list(KFold(5, shuffle=True, random_state=seed).split(ws.X))
    if protocol == "record":
        rng = np.random.RandomState(seed)
        recs = np.unique(ws.record); rng.shuffle(recs)
        grp = {r: i % 5 for i, r in enumerate(recs)}
        g = np.array([grp[r] for r in ws.record])
        return [(np.flatnonzero(g != k), np.flatnonzero(g == k)) for k in range(5)]
    if protocol == "subject":
        return list(LeaveOneGroupOut().split(ws.X, ws.y, ws.subject))
    raise ValueError(protocol)


def bigideas_folds(ws, protocol, seed):
    if protocol == "reading":
        return list(GroupKFold(5).split(ws.X, ws.y, ws.record))      # both windows of a reading stay together
    if protocol == "chrono0":
        return [chronological_split(ws, gap_s=0.0)]
    if protocol == "chrono":
        return [chronological_split(ws, gap_s=3600.0)]
    raise ValueError(protocol)


def load_derivative(path=None, win=320, hop=160):
    """Windows cut from the Kaggle derivative clean-dataset.csv exactly as the
    first draft did (consecutive rows, label = mean Glucose_level over the
    window).  Because the label cycles row by row, every window's label is
    the subject mean -- this rung reproduces the artefact under a CORRECT
    subject-level split."""
    import pandas as pd
    from ppg_data import WindowSet, _zscore_rows
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Archive_Dataset", "clean-dataset.csv")
    df = pd.read_csv(path, usecols=["PPG_Signal", "Patient_Id", "Glucose_level"])
    Xs, ys, subj, rec, meta = [], [], [], [], []
    for pid, g in df.groupby("Patient_Id", sort=True):
        x = g.PPG_Signal.to_numpy(np.float32); y = g.Glucose_level.to_numpy(np.float32)
        n = 1 + (len(x) - win) // hop
        for i in range(n):
            seg = x[i * hop:i * hop + win]
            if seg.std() < 1e-6:
                continue                                        # the ~26 % flat windows carry no signal at all
            Xs.append(seg); ys.append(y[i * hop:i * hop + win].mean())
            subj.append(f"{int(pid):02d}"); rec.append(f"{int(pid):02d}_{i:05d}")
        meta.append({"record": f"{int(pid):02d}", "subject": f"{int(pid):02d}", "glucose": float(y.mean()),
                     "n_labels": int(len(np.unique(y)))})
    ws = WindowSet(_zscore_rows(np.stack(Xs)), np.array(ys, np.float32), np.array(subj), np.array(rec),
                   pd.DataFrame(meta), source="Kaggle derivative")
    log.info(ws.describe())
    return ws


def derivative_folds(ws, protocol, seed):
    """'record': 5-fold over windows (subjects shared) -- the regime in which the
    collapsed labels can be memorised, i.e. the oracle rung; 'subject': LOSO."""
    if protocol == "record":
        return list(KFold(5, shuffle=True, random_state=seed).split(ws.X))
    if protocol == "subject":
        return list(LeaveOneGroupOut().split(ws.X, ws.y, ws.subject))
    raise ValueError(protocol)


def leaky_subject_mean(ws, tr, te):
    """Test window predicted with the mean of its OWN subject's training labels
    (falls back to the global train mean for subjects absent from training)."""
    pred = np.full(len(te), ws.y[tr].mean(), np.float32)
    for s in np.unique(ws.subject[te]):
        m = ws.subject[tr] == s
        if m.any():
            pred[ws.subject[te] == s] = ws.y[tr][m].mean()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mendeley", "bigideas", "derivative"], default="mendeley")
    ap.add_argument("--models", nargs="+", default=["mean", "subjmean", "1d"],
                    choices=["mean", "subjmean", "fusion", "1d", "2d", "3d"])
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dataset == "mendeley":
        ws, folds_fn = load_mendeley(), mendeley_folds
        protocols = args.protocols or ["window", "record", "subject"]
        epochs, batch = args.epochs or 40, args.batch or 16
    elif args.dataset == "derivative":
        ws, folds_fn = load_derivative(), derivative_folds
        protocols = args.protocols or ["record", "subject"]
        epochs, batch = args.epochs or 15, args.batch or 32
    else:
        ws, folds_fn = load_bigideas(max_windows_per_egv=2), bigideas_folds
        protocols = args.protocols or ["reading", "chrono0", "chrono"]
        epochs, batch = args.epochs or 15, args.batch or 64
    views = make_views(ws.X)
    dev = device()
    path = os.path.join(OUT_DIR, "leakage_ladder.json")
    ladder = json.load(open(path)) if os.path.exists(path) else {}

    for protocol in protocols:
        for name in args.models:
            for seed in (args.seeds if name in NETS else [0]):
                folds = folds_fn(ws, protocol, seed)
                pred = np.full(len(ws), np.nan, np.float32)
                t0 = time.time()
                for tr, te in folds:
                    if name in NETS:
                        pred[te] = train_fold(name, views, ws.y, tr, te, epochs, batch, 1e-3, seed, dev)
                    elif name == "subjmean":
                        pred[te] = leaky_subject_mean(ws, tr, te)
                    elif args.dataset == "bigideas" and name == "mean":
                        pred[te] = subject_mean_baseline(ws, tr, te) if protocol != "reading" else leaky_subject_mean(ws, tr, te)
                    else:
                        pred[te] = np.full(len(te), ws.y[tr].mean(), np.float32)
                keep = ~np.isnan(pred)
                m = metrics(recording_level(pred[keep], _subset(ws, keep)))
                m.update(dataset=args.dataset, protocol=protocol, model=name, seed=seed, epochs=epochs,
                         seconds=round(time.time() - t0, 1))
                merge_json(path, f"{args.dataset}/{protocol}/{name}/seed{seed}", m)
                log.info("LADDER %-8s %-8s %-8s seed %d  MAE %.2f  r %.3f  zoneA %.1f  (%.0f s)", args.dataset,
                         protocol, name, seed, m["mae"], m.get("pearson_r", float("nan")), m.get("zone_A_pct", float("nan")),
                         m["seconds"])


if __name__ == "__main__":
    main()
