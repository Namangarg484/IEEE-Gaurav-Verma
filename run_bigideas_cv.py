"""
run_bigideas_cv.py
==================
Within-subject glucose tracking on the PhysioNet BIG IDEAs dataset
(Empatica E4 BVP at 64 Hz vs. Dexcom G6 CGM), the only open dataset with
genuine glycaemic dynamics inside each subject.

Protocols
---------
within : per subject, CHRONOLOGICAL split -- the first 70 % of CGM readings
         train, the last 30 % test, with a 1-h guard gap so autocorrelated
         windows never straddle the split.  One population model is trained
         on all subjects' train portions; metrics are reported per subject
         and pooled.  Baseline "mean" is each subject's own train-period mean
         (the right null model for a personalised sensor).
loso   : leave-one-subject-out across subjects (cross-person generalisation).

Usage
-----
    python run_bigideas_cv.py --protocol within --models mean ridge fusion 1d
    python run_bigideas_cv.py --protocol loso --subjects 001 002 003 --quick

Outputs (results/)
------------------
    bigideas_<protocol>_<model>_seed<k>.csv   out-of-sample CGM-level predictions
    bigideas_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from ppg_data import handcrafted_features, load_bigideas, make_views
from run_subject_cv import (NETS, OUT_DIR, device, merge_json, metrics, predict_classical, recording_level,
                            train_fold, _subset)

log = logging.getLogger("bigideas_cv")


def subject_train_means(ws, tr: np.ndarray) -> dict:
    """Each subject's mean glucose over its TRAINING windows only."""
    return {s: float(ws.y[tr][ws.subject[tr] == s].mean()) for s in np.unique(ws.subject[tr])}


def time_of_day_baseline(ws, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Ridge on sin/cos of local hour plus a per-subject offset (train-side only)."""
    from sklearn.linear_model import RidgeCV
    t = ws.meta.set_index("record").egv_time.loc[ws.record].to_numpy()
    hour = (t % 86400) / 3600.0
    subs = np.unique(ws.subject[tr])
    def design(idx):
        h = hour[idx]
        onehot = (ws.subject[idx][:, None] == subs[None, :]).astype(np.float32)
        return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                                np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24), onehot])
    m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(design(tr), ws.y[tr])
    return m.predict(design(te)).astype(np.float32)


def persistence_baseline(ws, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Last glucose value of the subject's training period (a 'spot-check' null)."""
    t = ws.meta.set_index("record").egv_time.loc[ws.record].to_numpy()
    pred = np.empty(len(te), np.float32)
    for s in np.unique(ws.subject[te]):
        m = ws.subject[tr] == s
        last = tr[m][np.argmax(t[tr][m])]
        pred[ws.subject[te] == s] = ws.y[last]
    return pred


def chronological_split(ws, train_frac: float = 0.7, gap_s: float = 3600.0):
    """Return (train_idx, test_idx) over windows, split per subject by CGM time."""
    rec_time = ws.meta.set_index("record").egv_time
    t = rec_time.loc[ws.record].to_numpy()
    tr, te = np.zeros(len(ws), bool), np.zeros(len(ws), bool)
    for s in np.unique(ws.subject):
        m = ws.subject == s
        cut = np.quantile(np.unique(t[m]), train_frac)
        tr |= m & (t < cut - gap_s)
        te |= m & (t >= cut)
    return np.flatnonzero(tr), np.flatnonzero(te)


def subject_mean_baseline(ws, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    pred = np.empty(len(te), np.float32)
    for s in np.unique(ws.subject[te]):
        pred[ws.subject[te] == s] = ws.y[tr][ws.subject[tr] == s].mean()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=["within", "personal", "loso"], default="within",
                    help="within: one population model on all subjects' first 70%%; personal: one model per subject "
                         "trained on that subject's own first 70%%; loso: leave-one-subject-out")
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "fusion"],
                    choices=list(NETS) + ["mean", "ridge", "gbm", "tod", "persist"])
    ap.add_argument("--target", choices=["absolute", "residual"], default="absolute",
                    help="residual: networks/ridge predict y minus the subject's TRAINING mean and the mean is "
                         "added back (a personalised arm that gives the models the same information as the null)")
    ap.add_argument("--train-frac", type=float, default=0.7, help="chronological cut (fraction of readings in train)")
    ap.add_argument("--gap-s", type=float, default=3600.0, help="guard gap removed from the training side (s)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-egv", type=int, default=2, help="5-s windows kept before each CGM reading")
    ap.add_argument("--quick", action="store_true", help="3 epochs, first 2 LOSO folds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.quick:
        args.epochs = 3

    ws = load_bigideas(subjects=args.subjects, max_windows_per_egv=args.max_per_egv)
    t0 = time.time()
    views = make_views(ws.X)
    feats = handcrafted_features(ws.X)
    log.info("views computed in %.0f s for %d windows", time.time() - t0, len(ws))

    if args.protocol == "within":
        folds = [chronological_split(ws, args.train_frac, args.gap_s)]
    elif args.protocol == "personal":
        tr_all, te_all = chronological_split(ws, args.train_frac, args.gap_s)
        folds = [(tr_all[ws.subject[tr_all] == s], te_all[ws.subject[te_all] == s]) for s in np.unique(ws.subject)]
        folds = [(tr, te) for tr, te in folds if len(tr) > 64 and len(te)]
    else:
        folds = list(LeaveOneGroupOut().split(ws.X, ws.y, ws.subject))
        if args.quick:
            folds = folds[:2]
    dev = device()
    summary_path = os.path.join(OUT_DIR, "bigideas_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for name in args.models:
        for seed in (args.seeds if name in NETS else [0]):
            pred = np.full(len(ws), np.nan, np.float32)
            t0 = time.time()
            for k, (tr, te) in enumerate(folds):
                residual = args.target == "residual" and args.protocol != "loso" and name not in ("mean", "tod", "persist")
                if residual:
                    mu = subject_train_means(ws, tr)
                    off_tr = np.array([mu[s] for s in ws.subject[tr]], np.float32)
                    off_te = np.array([mu.get(s, ws.y[tr].mean()) for s in ws.subject[te]], np.float32)
                    y_fit = ws.y.copy(); y_fit[tr] = ws.y[tr] - off_tr
                else:
                    y_fit, off_te = ws.y, 0.0
                if name in NETS:
                    pred[te] = train_fold(name, views, y_fit, tr, te, args.epochs, args.batch, args.lr, seed, dev) + off_te
                elif name == "mean" and args.protocol in ("within", "personal"):
                    pred[te] = subject_mean_baseline(ws, tr, te)
                elif name == "tod":
                    pred[te] = time_of_day_baseline(ws, tr, te)
                elif name == "persist":
                    pred[te] = persistence_baseline(ws, tr, te)
                else:
                    pred[te] = predict_classical(name, feats, None, y_fit, tr, te) + off_te
                log.info("%-7s seed %d fold %d/%d  n_train %d n_test %d  MAE(win) %.1f", name, seed, k + 1,
                         len(folds), len(tr), len(te), np.abs(pred[te] - ws.y[te]).mean())
            keep = ~np.isnan(pred)
            df = recording_level(pred[keep], _subset(ws, keep))
            m = metrics(df)
            m["per_subject"] = {s: metrics(g) for s, g in df.groupby("subject") if len(g) > 5}
            m.update(model=name, seed=seed, protocol=args.protocol, epochs=args.epochs, target=args.target,
                     train_frac=args.train_frac, gap_s=args.gap_s,
                     subjects=sorted(np.unique(ws.subject).tolist()), seconds=round(time.time() - t0, 1))
            suffix = ("" if args.target == "absolute" else "_res") + ("" if args.train_frac == 0.7 else f"_cut{int(args.train_frac * 100)}") \
                     + ("" if args.gap_s == 3600.0 else f"_gap{int(args.gap_s)}")
            tag = f"{args.protocol}_{name}{suffix}_seed{seed}"
            df.to_csv(os.path.join(OUT_DIR, f"bigideas_{tag}.csv"), index=False)
            merge_json(summary_path, tag, m)
            log.info("RESULT %-20s MAE %.2f RMSE %.2f MARD %.1f%% R2 %+.3f r %.3f zoneA %.1f%% (%.0f s)", tag,
                     m["mae"], m["rmse"], m["mard_pct"], m["r2"], m["pearson_r"], m["zone_A_pct"], m["seconds"])


if __name__ == "__main__":
    main()
