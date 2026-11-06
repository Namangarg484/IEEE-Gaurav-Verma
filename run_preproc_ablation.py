"""
run_preproc_ablation.py
=======================
Is the null result about the PPG channel or about the representation?  The
common representation (5-s windows, 0.5-8 Hz, per-window z-score) discards
DC/perfusion amplitude, vasomotion (< 0.5 Hz) and heart-rate variability.
This script re-runs the BIG IDEAs guarded chronological split with the 1-D
CNN and ridge under alternative representations:

    window length   5 / 10 / 30 / 60 s (one window per CGM reading, ending at the reading)
    band            0.5-8 Hz (default)  vs  0.1-12 Hz
    amplitude       per-window z-score (default)  vs  per-subject scale kept
    ridge features  + log raw-window std, + HRV proxy (SDNN of peak intervals) for >= 30-s windows

Usage
-----
    python run_preproc_ablation.py --configs default w10 w30 w60 band012 amp --models mean ridge 1d --seeds 0

Writes results/preproc_ablation.json (tag: <config>_<model>_seed<k>).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from scipy import signal as sps

from ppg_data import FS, handcrafted_features, load_bigideas, make_views
from run_bigideas_cv import chronological_split, subject_mean_baseline, subject_train_means
from run_subject_cv import OUT_DIR, device, metrics, predict_classical, recording_level, train_fold, _subset

log = logging.getLogger("preproc")

CONFIGS = {
    "default": dict(win_s=5, band=None, keep_amplitude=False),
    "w10": dict(win_s=10, band=None, keep_amplitude=False),
    "w30": dict(win_s=30, band=None, keep_amplitude=False),
    "w60": dict(win_s=60, band=None, keep_amplitude=False),
    "band012": dict(win_s=5, band=(0.1, 12.0), keep_amplitude=False),
    "amp": dict(win_s=5, band=None, keep_amplitude=True),
    "w60amp": dict(win_s=60, band=(0.1, 12.0), keep_amplitude=True),
}


def hrv_features(X: np.ndarray) -> np.ndarray:
    """SDNN / RMSSD proxies from systolic-peak intervals (needs >= 30 s)."""
    out = np.zeros((len(X), 2), np.float32)
    for i, x in enumerate(X):
        pk, _ = sps.find_peaks(x, distance=int(0.33 * FS), prominence=0.3 * x.std())
        if len(pk) > 4:
            ibi = np.diff(pk) / FS * 1000.0
            out[i] = [ibi.std(), np.sqrt(np.mean(np.diff(ibi) ** 2))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "1d"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--target", choices=["absolute", "residual"], default="residual")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    path = os.path.join(OUT_DIR, "preproc_ablation.json")
    store = json.load(open(path)) if os.path.exists(path) else {}
    dev = device()

    for cfg in args.configs:
        c = CONFIGS[cfg]
        win = int(c["win_s"] * FS)
        ws = load_bigideas(win=win, max_windows_per_egv=1, lookback_s=60.0, band=c["band"], keep_amplitude=c["keep_amplitude"])
        views = make_views(ws.X, which=("x1",))
        feats = handcrafted_features(ws.X)
        extra = [np.log(ws.meta.set_index("record").loc[ws.record, "raw_std"].to_numpy(np.float32) + 1e-6)[:, None]]
        if c["win_s"] >= 30:
            extra.append(hrv_features(ws.X))
        feats = np.hstack([feats] + extra)
        tr, te = chronological_split(ws)
        mu = subject_train_means(ws, tr)
        off_tr = np.array([mu[s] for s in ws.subject[tr]], np.float32)
        off_te = np.array([mu.get(s, ws.y[tr].mean()) for s in ws.subject[te]], np.float32)
        for name in args.models:
            for seed in (args.seeds if name == "1d" else [0]):
                t0 = time.time()
                pred = np.full(len(ws), np.nan, np.float32)
                residual = args.target == "residual" and name != "mean"
                y_fit = ws.y.copy()
                if residual:
                    y_fit[tr] = ws.y[tr] - off_tr
                if name == "mean":
                    pred[te] = subject_mean_baseline(ws, tr, te)
                elif name == "1d":
                    pred[te] = train_fold("1d", views, y_fit, tr, te, args.epochs, 64, 1e-3, seed, dev) + (off_te if residual else 0)
                else:
                    pred[te] = predict_classical(name, feats, None, y_fit, tr, te) + (off_te if residual else 0)
                keep = ~np.isnan(pred)
                m = metrics(recording_level(pred[keep], _subset(ws, keep)))
                m.update(config=cfg, model=name, seed=seed, target=args.target, win_s=c["win_s"], band=c["band"],
                         keep_amplitude=c["keep_amplitude"], n_test=int(len(te)), seconds=round(time.time() - t0, 1))
                store[f"{cfg}_{name}_seed{seed}"] = m
                json.dump(store, open(path, "w"), indent=2)
                log.info("PREPROC %-8s %-6s seed %d  MAE %.2f  r %.3f  within-r %s  (%.0f s)", cfg, name, seed, m["mae"],
                         m.get("pearson_r", float("nan")), m.get("within_subject_r_median"), m["seconds"])


if __name__ == "__main__":
    main()
