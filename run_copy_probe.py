"""
run_copy_probe.py
=================
Validation of the copy probe on a leak of known type: the Kermani-Esmaeili
leakage ladder, whose random-window rung places overlapping (time-shifted)
windows of the same recording on both sides of the split.

Two probe distances on z-scored 5-s windows:
    euclid  plain Euclidean distance (misses time-shifted copies)
    shift   1 - max normalised cross-correlation over all lags (shift-invariant)

Rungs: window_random (overlap copies cross the split), record_random (other
recordings of the same subject cross; no copies), subject (honest).  Window
predictions are averaged per recording and compared with the train-mean null
by the subject-cluster bootstrap.

    python run_copy_probe.py          # -> results/copy_probe.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from ppg_data import load_mendeley
from validate_detectors import cluster_ci

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "copy_probe.json")


def shift_distance(A, B):
    """1 - max over lags of the normalised circular cross-correlation (rows z-scored)."""
    n = A.shape[1]
    FA, FB = np.fft.rfft(A, 2 * n), np.fft.rfft(B, 2 * n)
    D = np.empty((len(A), len(B)))
    for i in range(len(A)):
        cc = np.fft.irfft(FA[i][None] * np.conj(FB), 2 * n)        # (len(B), 2n) linear cross-correlation
        D[i] = 1.0 - cc.max(1) / n
    return D


def probe(X, y, tr, te, kind):
    if kind == "euclid":
        D = ((X[te][:, None, :] - X[tr][None]) ** 2).sum(-1)
    else:
        D = shift_distance(X[te], X[tr])
    return y[tr][D.argmin(1)]


def main() -> None:
    ws = load_mendeley()
    X = (ws.X - ws.X.mean(1, keepdims=True)) / (ws.X.std(1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0)
    out = {}
    rungs = {"window_random": list(KFold(10, shuffle=True, random_state=0).split(X)),
             "record_random": list(GroupKFold(10).split(X, ws.y, ws.record)),
             "subject": list(GroupKFold(10).split(X, ws.y, ws.subject))}
    for rung, folds in rungs.items():
        for kind in ("euclid", "shift"):
            p = np.empty(len(ws)); pn = np.empty(len(ws))
            for tr, te in folds:
                p[te] = probe(X, ws.y, tr, te, kind); pn[te] = ws.y[tr].mean()
            r = (pd.DataFrame({"record": ws.record, "subject": ws.subject, "y": ws.y, "p": p, "pn": pn})
                 .groupby("record", as_index=False).agg(subject=("subject", "first"), y=("y", "first"), p=("p", "mean"), pn=("pn", "mean")))
            d, ci = cluster_ci(r.y.to_numpy(), r.p.to_numpy(), r.pn.to_numpy(), r.subject.astype(str).to_numpy(), rng, B=5000)
            out[f"{rung}_{kind}"] = {"mae": float(np.abs(r.p - r.y).mean()), "null_mae": float(np.abs(r.pn - r.y).mean()),
                                     "diff_vs_null": d, "diff_ci": [float(ci[0]), float(ci[1])], "flags": bool(ci[1] < 0)}
            print(f"{rung:14s} {kind:6s} MAE {out[f'{rung}_{kind}']['mae']:.2f} null {out[f'{rung}_{kind}']['null_mae']:.2f} "
                  f"delta {d:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
