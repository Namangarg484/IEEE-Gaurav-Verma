"""
run_subject_cv.py
=================
Subject-wise cross-validation of the 1D+2D+3D fusion network (and its ablations
and classical baselines) on the raw Mendeley PPG-glucose recordings.

Protocol (no leakage, no peeking)
---------------------------------
* Leave-one-subject-out (LOSO, 23 folds) or GroupKFold by subject.
* A fresh model per fold, fixed epoch budget, cosine LR -- no early stopping on
  the held-out subject.
* Glucose standardised with TRAIN statistics only; predictions de-standardised.
* Window predictions are averaged per recording (one reference value each),
  and every metric is reported at recording level on the out-of-fold
  predictions (n = 67).

Usage
-----
    python run_subject_cv.py                        # fusion model, LOSO
    python run_subject_cv.py --models fusion 1d 2d 3d mean ridge demo --seeds 0 1 2
    python run_subject_cv.py --folds 5 --epochs 30 --quick

Outputs (results/)
------------------
    subject_cv_<model>_seed<k>.csv    out-of-fold recording-level predictions
    subject_cv_summary.json           metrics for every (model, seed)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import HistGradientBoostingRegressor

from baseline_models import BASELINES
from fusion_network_model import Branch1D, Branch2D, Branch3D, MultimodalFusionNet
from plot_clarke_error_grid import summarize_clarke_zones
from ppg_data import handcrafted_features, load_mendeley, make_views

log = logging.getLogger("subject_cv")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results")
NETS = ("fusion", "1d", "2d", "3d") + tuple(BASELINES)
CLASSICAL = ("mean", "ridge", "demo", "ridge+demo", "gbm", "gbm+demo")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SingleBranch(nn.Module):
    """One branch + the same regression head as the fusion net (ablation)."""

    def __init__(self, which: str) -> None:
        super().__init__()
        self.which = which
        self.branch = {"1d": Branch1D, "2d": Branch2D, "3d": Branch3D}[which]()
        self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x1, x2, x3):
        return self.head(self.branch({"1d": x1, "2d": x2, "3d": x3}[self.which]))


def build_model(name: str) -> nn.Module:
    if name == "fusion":
        return MultimodalFusionNet()
    if name in BASELINES:
        return BASELINES[name]()
    return SingleBranch(name)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Training of one fold
# --------------------------------------------------------------------------- #
def train_fold(name: str, views: dict, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               epochs: int, batch: int, lr: float, seed: int, dev: torch.device) -> np.ndarray:
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(name).to(dev)
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-6)
    yt = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    xs = {k: v[tr] for k, v in views.items()}
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for b in range(len(tr) // batch):                 # drop_last: BatchNorm needs > 1 sample
            idx = perm[b * batch:(b + 1) * batch]
            pred = model(xs["x1"][idx].to(dev), xs["x2"][idx].to(dev), xs["x3"][idx].to(dev)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, yt[idx].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        out = []
        for b in range(0, len(te), 256):
            i = te[b:b + 256]
            out.append(model(views["x1"][i].to(dev), views["x2"][i].to(dev), views["x3"][i].to(dev)).squeeze(1).cpu())
    return torch.cat(out).numpy() * sd + mu


def predict_classical(name: str, feats: np.ndarray, demo: np.ndarray, y: np.ndarray,
                      tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    if name == "mean":
        return np.full(len(te), y[tr].mean(), np.float32)
    F = feats if name in ("ridge", "gbm") else demo if name == "demo" else np.hstack([feats, demo])
    if name.startswith("gbm"):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=10, l2_regularization=1.0, random_state=0)
        return m.fit(F[tr], y[tr]).predict(F[te]).astype(np.float32)
    sc = StandardScaler().fit(F[tr])
    m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(F[tr]), y[tr])
    return m.predict(sc.transform(F[te])).astype(np.float32)


# --------------------------------------------------------------------------- #
# Metrics at recording level
# --------------------------------------------------------------------------- #
def recording_level(pred: np.ndarray, ws) -> pd.DataFrame:
    df = pd.DataFrame({"record": ws.record, "subject": ws.subject, "y": ws.y, "pred": pred})
    return df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"),
                                                                 n_windows=("pred", "size"))


def metrics(df: pd.DataFrame) -> dict:
    y, p = df.y.to_numpy(), df.pred.to_numpy()
    if len(y) < 2 or y.std() == 0:
        return {"n": int(len(y)), "mae": float(np.abs(p - y).mean()) if len(y) else None, "error": "degenerate split"}
    e = p - y
    ss_res, ss_tot = float((e ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    cz = summarize_clarke_zones(y.astype(np.float32), p.astype(np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        within = df.groupby("subject").apply(
            lambda g: np.corrcoef(g.y, g.pred)[0, 1] if len(g) > 2 and g.y.std() > 0 and g.pred.std() > 0 else np.nan,
            include_groups=False)
    return {"n": int(len(df)), "mae": float(np.abs(e).mean()), "rmse": float(math.sqrt((e ** 2).mean())),
            "mard_pct": float((np.abs(e) / y).mean() * 100), "r2": 1 - ss_res / ss_tot,
            "pearson_r": float(np.corrcoef(y, p)[0, 1]),
            "within_subject_r_median": float(np.nanmedian(within)) if np.isfinite(within).any() else None,
            **{f"zone_{z}_pct": cz[f"zone_{z}_pct"] for z in "ABCDE"}}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["fusion"], choices=list(NETS) + list(CLASSICAL))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", default="loso", help="'loso' or an integer for GroupKFold")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true", help="only the first 3 folds (smoke test)")
    ap.add_argument("--dataset", choices=["mendeley", "stuba"], default="mendeley")
    ap.add_argument("--channel", default="IR1", help="STUBA PPG channel (R1/IR1/G1/R2/IR2/G2/mean)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dataset == "stuba":
        from ppg_data import load_stuba
        ws = load_stuba(channel=args.channel)
        prefix = f"stuba_{args.channel}_"
    else:
        ws = load_mendeley()
        prefix = ""
    if args.folds != "loso":
        prefix = f"kfold{int(args.folds)}_" + prefix            # never overwrite the LOSO result files
    t0 = time.time()
    views = make_views(ws.X)
    feats = handcrafted_features(ws.X)
    demo = ws.meta.set_index("record").loc[ws.record, ["age", "height", "weight"]].to_numpy(np.float32)
    demo = np.hstack([demo, (ws.meta.set_index("record").loc[ws.record, "gender"] == "Male").to_numpy()[:, None]])
    log.info("views computed in %.1f s: %s", time.time() - t0, {k: tuple(v.shape) for k, v in views.items()})

    splitter = LeaveOneGroupOut() if args.folds == "loso" else GroupKFold(int(args.folds))
    folds = list(splitter.split(ws.X, ws.y, ws.subject))
    if args.quick:
        folds = folds[:3]
    dev = device()
    summary_path = os.path.join(OUT_DIR, "subject_cv_summary.json" if args.dataset == "mendeley" else "stuba_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for name in args.models:
        for seed in (args.seeds if name in NETS else [0]):
            pred = np.full(len(ws), np.nan, np.float32)
            t0 = time.time()
            for k, (tr, te) in enumerate(folds):
                if name in NETS:
                    pred[te] = train_fold(name, views, ws.y, tr, te, args.epochs, args.batch, args.lr, seed, dev)
                else:
                    pred[te] = predict_classical(name, feats, demo, ws.y, tr, te)
                log.info("%-10s seed %d fold %2d/%d held-out %s  MAE(win) %.1f", name, seed, k + 1, len(folds),
                         np.unique(ws.subject[te]), np.abs(pred[te] - ws.y[te]).mean())
            keep = ~np.isnan(pred)
            df = recording_level(pred[keep], _subset(ws, keep))
            m = metrics(df)
            m.update(model=name, seed=seed, folds=args.folds, epochs=args.epochs, seconds=round(time.time() - t0, 1))
            tag = f"{prefix}{name}_seed{seed}"
            df.to_csv(os.path.join(OUT_DIR, f"subject_cv_{tag}.csv"), index=False)
            merge_json(summary_path, tag, m)
            log.info("RESULT %-12s MAE %.2f  RMSE %.2f  R2 %+.3f  r %.3f  zoneA %.1f%%  (%.0f s)", tag, m["mae"],
                     m["rmse"], m["r2"], m["pearson_r"], m["zone_A_pct"], m["seconds"])


def merge_json(path: str, key: str, value: dict) -> None:
    """Re-read the store and write back with this entry, so concurrent runs
    writing to the same summary file do not clobber each other."""
    store = json.load(open(path)) if os.path.exists(path) else {}
    store[key] = value
    tmp = path + ".tmp"
    json.dump(store, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


def _subset(ws, mask):
    from copy import copy
    s = copy(ws)
    s.X, s.y, s.subject, s.record = ws.X[mask], ws.y[mask], ws.subject[mask], ws.record[mask]
    return s


if __name__ == "__main__":
    main()
