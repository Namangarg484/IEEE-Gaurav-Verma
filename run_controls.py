"""
run_controls.py
===============
Positive and negative controls for the benchmark, run through the IDENTICAL
windows, views, model and split protocol as the glucose experiments.  A
negative-result paper needs these to show that "no model beats the mean" is a
statement about the signal, not about a broken pipeline.

Controls (Mendeley, leave-one-subject-out, 1-D CNN unless stated)
    hr        regress the window's own heart rate (Welch peak, bpm).  Must be
              recoverable: Pearson r > 0.9 expected.
    age       regress subject age under LOSO (weak but non-zero expected).
    perm      glucose labels permuted ACROSS RECORDINGS (a fixed permutation
              per seed).  Must return to the null model.
    noise     white-noise input with the real labels.  Must return to the null.
    glucose   the real task, for side-by-side reporting.

Controls (BIG IDEAs, chronological split with 1-h guard gap)
    hr        as above (must be recoverable).
    perm      labels permuted across CGM readings within each subject.

Usage
-----
    python run_controls.py --dataset mendeley --controls hr age perm noise glucose --seeds 0 1 2
    python run_controls.py --dataset bigideas --controls hr perm --seeds 0

Writes results/controls.json and results/controls_<dataset>_<control>_seed<k>.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut

from ppg_data import handcrafted_features, load_bigideas, load_mendeley, make_views
from run_bigideas_cv import chronological_split, subject_mean_baseline
from run_subject_cv import OUT_DIR, device, metrics, recording_level, train_fold, _subset

log = logging.getLogger("controls")


def targets_for(ws, control: str, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if control == "glucose":
        return ws.y.astype(np.float32)
    if control == "hr":                                                       # zero-padded FFT peak in 0.7-3 Hz (bpm)
        nfft = 8192
        spec = np.abs(np.fft.rfft(ws.X * np.hanning(ws.X.shape[1]), n=nfft, axis=1))
        f = np.fft.rfftfreq(nfft, 1.0 / 64)
        band = (f >= 0.7) & (f <= 3.0)
        return (f[band][spec[:, band].argmax(1)] * 60.0).astype(np.float32)
    if control == "age":
        return ws.meta.set_index("record").loc[ws.record, "age"].to_numpy(np.float32)
    if control == "perm":                                                     # permute labels across records
        recs = np.unique(ws.record)
        rec_y = {r: float(ws.y[ws.record == r][0]) for r in recs}
        shuffled = dict(zip(recs, rng.permutation([rec_y[r] for r in recs])))
        return np.array([shuffled[r] for r in ws.record], np.float32)
    raise ValueError(control)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mendeley", "bigideas"], default="mendeley")
    ap.add_argument("--controls", nargs="+", default=["hr", "age", "perm", "noise", "glucose"])
    ap.add_argument("--model", default="1d")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dataset == "mendeley":
        ws = load_mendeley(); epochs = args.epochs or 40; batch = 16
        folds = list(LeaveOneGroupOut().split(ws.X, ws.y, ws.subject))
    else:
        ws = load_bigideas(max_windows_per_egv=2); epochs = args.epochs or 15; batch = 64
        folds = [chronological_split(ws)]
    if args.quick:
        folds, epochs = folds[:3], 2
    views = make_views(ws.X)
    dev = device()
    path = os.path.join(OUT_DIR, "controls.json")
    store = json.load(open(path)) if os.path.exists(path) else {}

    for control in args.controls:
        for seed in args.seeds:
            y = targets_for(ws, "glucose" if control == "noise" else control, seed)
            v = views
            if control == "noise":
                g = torch.Generator().manual_seed(seed)
                v = {k: torch.randn(t.shape, generator=g) for k, t in views.items()}
            pred = np.full(len(ws), np.nan, np.float32)
            null = np.full(len(ws), np.nan, np.float32)
            t0 = time.time()
            for tr, te in folds:
                pred[te] = train_fold(args.model, v, y, tr, te, epochs, batch, 1e-3, seed, dev)
                null[te] = (subject_mean_baseline(_subset_y(ws, y), tr, te) if args.dataset == "bigideas"
                            else np.full(len(te), y[tr].mean(), np.float32))
            keep = ~np.isnan(pred)
            wsy = _subset_y(ws, y)
            m = metrics(recording_level(pred[keep], _subset(wsy, keep)))
            m_null = metrics(recording_level(null[keep], _subset(wsy, keep)))
            m.update(control=control, dataset=args.dataset, model=args.model, seed=seed, epochs=epochs,
                     null_mae=m_null.get("mae"), null_r=m_null.get("pearson_r"), seconds=round(time.time() - t0, 1),
                     target_mean=float(y.mean()), target_std=float(y.std()))
            tag = f"{args.dataset}_{control}_seed{seed}"
            recording_level(pred[keep], _subset(wsy, keep)).to_csv(os.path.join(OUT_DIR, f"controls_{tag}.csv"), index=False)
            store[tag] = m
            json.dump(store, open(path, "w"), indent=2)
            log.info("CONTROL %-22s MAE %.2f (null %.2f)  r %.3f  target sd %.1f  (%.0f s)", tag, m["mae"],
                     m["null_mae"], m.get("pearson_r", float("nan")), y.std(), m["seconds"])


def _subset_y(ws, y):
    from copy import copy
    s = copy(ws); s.y = y.astype(np.float32)
    return s


if __name__ == "__main__":
    main()
