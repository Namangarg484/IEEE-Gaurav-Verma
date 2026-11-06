"""
run_reaudit.py
==============
Re-audit of a published PPG -> glucose result with the leakage audit.

Target claim: Soliman et al., "Non-Invasive Glucose Level Monitoring from PPG
using a Hybrid CNN-GRU Deep Learning Network", arXiv:2411.11094 (2024):
MAE 2.96 mg/dL, RMSE 3.94, R^2 0.97, 100 % Clarke A on the Kermani-Esmaeili
recordings (67 signals, 23 subjects).

What the paper states, and is reproduced exactly here
    * band-pass 0.5-8 Hz; downsampling 2175 -> 30 Hz;
    * Gaussian-noise augmentation of the signals, 67 -> 269 samples;
    * "we shuffle the dataset", then train / validation / test; 10-fold CV.
What it does not state, resolved both ways or by a stated default
    * whether augmentation precedes the split (protocol A) or is confined to
      the training folds (protocols B, C);
    * the split unit: samples (A), recordings (B) or subjects (C);
    * segmentation (none: one 10-s signal = one sample, 300 points at 30 Hz),
      normalisation (per-signal z-score), noise SDs (0.05/0.10/0.20 of the
      signal SD), layer sizes and training settings (see ``CnnGru``).

Protocols, all 10-fold
    A  augment all 67 signals -> 268 samples, shuffle, folds over samples
       (a test sample's noisy siblings and its other-noise copies are in
       training: copy leak + record leak + subject leak)
    B  folds over the 67 recordings; augment training recordings only
       (subject leak only: other recordings of the same person in training)
    C  folds over the 23 subjects; augment training subjects only (honest)

Models: train-mean null; 1-nearest-neighbour on the raw signal (a pure
memoriser -- it can only succeed by finding a copy); CNN-GRU re-implementation
following the paper's description (3 seeds).

    python run_reaudit.py              # -> results/reaudit_soliman.json
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
import torch.nn as nn
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.model_selection import GroupKFold, KFold

from ppg_data import MENDELEY_FS, MENDELEY_ROOT
from ppgbench.metrics import clarke_summary
from run_subject_cv import merge_json
from validate_detectors import cluster_ci

log = logging.getLogger("reaudit")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "reaudit_soliman.json")
FS_OUT, N_PTS, COPIES, NOISE = 30, 300, 3, (0.05, 0.10, 0.20)


def load_signals():
    X, y, subj, rec = [], [], [], []
    sos = butter(4, [0.5, 8.0], btype="band", fs=MENDELEY_FS, output="sos")
    for sf in sorted(glob.glob(os.path.join(MENDELEY_ROOT, "RawData", "signal_*.csv"))):
        key = os.path.basename(sf)[len("signal_"):-len(".csv")]
        raw = pd.read_csv(sf, header=None).to_numpy(np.float64).ravel()
        lab = pd.read_csv(os.path.join(MENDELEY_ROOT, "Labels", f"label_{key}.csv")).iloc[0]
        x = resample_poly(sosfiltfilt(sos, raw - raw.mean()), 2, 145)[:N_PTS]          # 2175 * 2/145 = 30 Hz
        X.append((x - x.mean()) / (x.std() + 1e-9)); y.append(float(lab["Glucose"]))
        subj.append(key.split("_")[0]); rec.append(key)
    return np.array(X, np.float32), np.array(y), np.array(subj), np.array(rec)


def augment(X, y, idx, rng, noise_scale=1.0):
    """Originals plus COPIES noisy versions of each signal in ``idx``; returns arrays + source index.
    67 originals + 3 x 67 copies = 268 samples (the paper reports 269 without giving the scheme)."""
    xs, ys, src = [X[idx]], [y[idx]], [idx]
    for sd in NOISE[:COPIES]:
        xs.append(X[idx] + rng.normal(0, sd * noise_scale, X[idx].shape).astype(np.float32)); ys.append(y[idx]); src.append(idx)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(src)


class CnnGru(nn.Module):
    """Three parallel branches (paper Sec. III-B6): two Conv1D blocks with different kernel sizes
    (conv -> BN -> ReLU -> max-pool) and one GRU block; each flattened and passed through three
    identical fully connected layers; concatenated; one linear regression unit.  Sizes are not
    given in the paper; the values below are stated defaults."""

    def __init__(self, n=N_PTS, filters=16, hidden=32, gru=32):
        super().__init__()
        def conv(k):
            return nn.Sequential(nn.Conv1d(1, filters, k, padding=k // 2), nn.BatchNorm1d(filters), nn.ReLU(), nn.MaxPool1d(4))
        def head(d):
            return nn.Sequential(nn.Flatten(), nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.c1, self.c2 = conv(5), conv(15)
        self.h1, self.h2 = head(filters * (n // 4)), head(filters * (n // 4))
        self.gru = nn.GRU(1, gru, batch_first=True)
        self.h3 = head(gru)
        self.out = nn.Linear(3 * hidden, 1)

    def forward(self, x):                              # x: (B, n)
        a = self.h1(self.c1(x[:, None])); b = self.h2(self.c2(x[:, None]))
        _, h = self.gru(x[:, ::3, None])               # GRU over the 10-Hz decimated sequence (speed; disclosed)
        return self.out(torch.cat([a, b, self.h3(h[-1])], 1)).squeeze(1)


def train_cnngru(Xtr, ytr, Xva, yva, seed, epochs=150):
    torch.manual_seed(seed)
    mu, sd = ytr.mean(), ytr.std() + 1e-9
    net = CnnGru(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, yt = torch.tensor(Xtr), torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    Xv = torch.tensor(Xva)
    best, best_state = np.inf, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(16):
            opt.zero_grad(); loss = nn.functional.mse_loss(net(Xt[b]), yt[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(np.abs(net(Xv).numpy() * sd + mu - yva).mean())
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    return lambda Z: net(torch.tensor(Z)).detach().numpy() * sd + mu


def one_nn(Xtr, ytr, Xte):
    d = ((Xte[:, None, :] - Xtr[None]) ** 2).sum(-1)
    return ytr[d.argmin(1)]


def run_protocol(proto, X, y, subj, rec, seed, noise_scale=1.0):
    """Returns a DataFrame of test predictions (one row per test sample).

    Early-stopping validation is drawn the way the protocol draws test units:
    A -> 10 % of the (augmented) training samples; B -> 10 % of the training
    *recordings*, C -> 10 % of the training *subjects*, both held out before
    augmentation, so no copy of a validation signal is ever trained on.
    """
    rng = np.random.default_rng(seed)
    rows = []
    if proto == "A":
        Xa, ya, src = augment(X, y, np.arange(len(X)), rng, noise_scale)
        folds = KFold(10, shuffle=True, random_state=seed).split(Xa)
    else:
        groups = rec if proto == "B" else subj
        folds = GroupKFold(10).split(X, y, groups) if proto == "C" else KFold(10, shuffle=True, random_state=seed).split(X)
    for fold, (tr, te) in enumerate(folds):
        vrng = np.random.default_rng(seed + fold)
        if proto == "A":
            vr = vrng.permutation(len(tr)); nv = max(1, len(tr) // 10)
            fit_i, val_i = tr[vr[nv:]], tr[vr[:nv]]
            Xf, yf, Xv, yv = Xa[fit_i], ya[fit_i], Xa[val_i], ya[val_i]
            Xtr, ytr = Xa[tr], ya[tr]
            Xte, yte, src_te = Xa[te], ya[te], src[te]
        else:
            unit = rec if proto == "B" else subj
            units_tr = np.unique(unit[tr]); nv = max(1, len(units_tr) // 10)
            val_units = set(vrng.choice(units_tr, nv, replace=False).tolist())
            is_val = np.array([unit[i] in val_units for i in tr])
            fit_i, val_i = tr[~is_val], tr[is_val]
            Xf, yf, _ = augment(X, y, fit_i, rng, noise_scale)
            Xv, yv = X[val_i], y[val_i]
            Xtr, ytr, _ = augment(X, y, tr, np.random.default_rng(seed + 1000 + fold), noise_scale)
            Xte, yte, src_te = X[te], y[te], te
        f = train_cnngru(Xf, yf, Xv, yv, seed)
        p_cg = f(Xte); p_nn = one_nn(Xtr, ytr, Xte); p_null = np.full(len(yte), ytr.mean())
        for i in range(len(yte)):
            rows.append({"fold": fold, "record": rec[src_te[i]], "subject": subj[src_te[i]], "y": yte[i],
                         "cnngru": float(p_cg[i]), "nn1": float(p_nn[i]), "null": float(p_null[i])})
    return pd.DataFrame(rows)


def summarise(df, model, rng):
    y, p = df.y.to_numpy(), df[model].to_numpy()
    e = p - y
    out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
           "mae_fold_mean": float(df.assign(ae=np.abs(e)).groupby("fold").ae.mean().mean()),
           "zone_A_pct": float(clarke_summary(y, p)["zone_A_pct"]), "n_test": int(len(df))}
    if model != "null":
        d, ci = cluster_ci(y, p, df.null.to_numpy(), df.subject.to_numpy(), rng)
        out.update(diff_vs_null=d, diff_ci=[float(ci[0]), float(ci[1])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--protocols", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiplies the augmentation noise SDs (sensitivity)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(2)
    X, y, subj, rec = load_signals()
    log.info("%d signals, %d subjects, %d points at %d Hz", len(X), len(set(subj)), X.shape[1], FS_OUT)
    rng = np.random.default_rng(0)
    for proto in args.protocols:
        for seed in args.seeds:
            t0 = time.time()
            df = run_protocol(proto, X, y, subj, rec, seed, args.noise_scale)
            tag = f"{proto}_seed{seed}" + ("" if args.noise_scale == 1.0 else f"_noise{args.noise_scale:g}")
            df.to_csv(os.path.join(ROOT, "results", f"reaudit_soliman_{tag}.csv"), index=False)
            res = {m: summarise(df, m, rng) for m in ("null", "nn1", "cnngru")}
            merge_json(OUT, tag, res)              # atomic re-read + merge (parallel runs)
            log.info("RESULT %s seed %d: null %.2f | 1-NN %.2f | CNN-GRU %.2f (R2 %.2f, zone A %.0f%%) [%.0f s]", proto, seed,
                     res["null"]["mae"], res["nn1"]["mae"], res["cnngru"]["mae"], res["cnngru"]["r2"],
                     res["cnngru"]["zone_A_pct"], time.time() - t0)


if __name__ == "__main__":
    main()
