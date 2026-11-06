"""
ppg_data.py
===========
Valid data loaders for the 1D+2D+3D PPG -> glucose fusion network.

This module replaces ``fusion_data_pipeline.py``, which read
``Archive_Dataset/clean-dataset.csv``.  That file is a corrupted cross-join
(every PPG sample paired with every glucose label of the same subject), so
any result trained on it is meaningless.  Everything here starts from raw
recordings instead.

Sources
-------
1. Mendeley "PPG dataset for blood glucose" (Archive_Dataset/archive (2)/PPG_Dataset)
   67 recordings from 23 subjects; each is 21,900 samples = 10 s at 2175 Hz
   with ONE reference glucose value (Labels/label_SS_RRRR.csv).
2. PhysioNet BIG IDEAs v1.1.3 (data.nosync/bigideas/sub_XXX.npz written by
   scripts/fetch_bigideas.py): Empatica E4 BVP at 64 Hz in 120-s windows centred
   on each Dexcom G6 reading.

Common representation
---------------------
Both sources are brought to 64 Hz, band-passed 0.5-8 Hz, cut into 5-s windows
(WIN = 320 samples) and z-scored per window.  The three model views
(1-D signal, CWT scalogram, Takens point cloud) are computed once by
``make_views`` so that cross-validation folds just index tensors.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
from scipy import signal as sps

log = logging.getLogger(__name__)

FS = 64                 # common sampling rate (Hz) == Empatica E4 BVP
WIN = 5 * FS            # 320 samples per window
HOP = WIN // 2          # 50 % overlap for the short Mendeley recordings
MENDELEY_FS = 2175
TAU = 5                 # Takens delay (samples)
PHASE_POINTS = 256
CWT_SCALES = 64
SPEC = 64               # scalogram side (pixels)

MENDELEY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Archive_Dataset", "archive (2)", "PPG_Dataset")
BIGIDEAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data.nosync", "bigideas")


# --------------------------------------------------------------------------- #
# Signal conditioning
# --------------------------------------------------------------------------- #
def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 8.0, order: int = 4) -> np.ndarray:
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x.astype(np.float64))


def to_common_rate(x: np.ndarray, fs_in: float, lo: float = 0.5, hi: float = 8.0) -> np.ndarray:
    """Band-pass at the native rate, then resample to FS (64 Hz)."""
    x = bandpass(x - np.mean(x), fs_in, lo, hi)
    g = np.gcd(int(fs_in), FS)
    return sps.resample_poly(x, FS // g, int(fs_in) // g).astype(np.float32)


def sliding_windows(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    n = 1 + (len(x) - win) // hop if len(x) >= win else 0
    return np.stack([x[i * hop:i * hop + win] for i in range(n)]) if n else np.empty((0, win), np.float32)


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class WindowSet:
    """Windows plus the labels/groups needed for subject-wise evaluation."""
    X: np.ndarray                      # (N, WIN) float32, z-scored
    y: np.ndarray                      # (N,) glucose mg/dL
    subject: np.ndarray                # (N,) str   -> split key (never leak a subject)
    record: np.ndarray                 # (N,) str   -> one reference measurement
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per record
    source: str = ""

    def __len__(self) -> int:
        return len(self.y)

    def describe(self) -> str:
        return (f"{self.source}: {len(self)} windows, {len(np.unique(self.record))} records, "
                f"{len(np.unique(self.subject))} subjects, glucose {self.y.min():.0f}-{self.y.max():.0f} "
                f"(mean {self.y.mean():.1f} +/- {self.y.std():.1f}) mg/dL")


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# Mendeley raw recordings
# --------------------------------------------------------------------------- #
def load_mendeley(root: str = MENDELEY_ROOT, win: int = WIN, hop: int = HOP) -> WindowSet:
    sig_files = sorted(glob.glob(os.path.join(root, "RawData", "signal_*.csv")))
    if not sig_files:
        raise FileNotFoundError(f"no signal_*.csv under {root}")
    Xs, ys, subj, rec, meta = [], [], [], [], []
    for sf in sig_files:
        key = os.path.basename(sf)[len("signal_"):-len(".csv")]        # SS_RRRR
        lf = os.path.join(root, "Labels", f"label_{key}.csv")
        if not os.path.exists(lf):
            log.warning("no label for %s; skipped", key)
            continue
        raw = pd.read_csv(sf, header=None).to_numpy(np.float64).ravel()
        lab = pd.read_csv(lf).iloc[0]
        x = to_common_rate(raw, MENDELEY_FS)
        W = sliding_windows(x, win, hop)
        Xs.append(W)
        ys.append(np.full(len(W), float(lab["Glucose"]), np.float32))
        subj.append(np.full(len(W), key.split("_")[0]))
        rec.append(np.full(len(W), key))
        meta.append({"record": key, "subject": key.split("_")[0], "glucose": float(lab["Glucose"]),
                     "gender": str(lab["Gender"]), "age": float(lab["Age"]),
                     "height": float(lab["Height"]), "weight": float(lab["Weight"]),
                     "raw_std": float(raw.std()), "n_windows": int(len(W))})
    ws = WindowSet(_zscore_rows(np.concatenate(Xs)), np.concatenate(ys), np.concatenate(subj),
                   np.concatenate(rec), pd.DataFrame(meta), source="Mendeley")
    log.info(ws.describe())
    return ws


# --------------------------------------------------------------------------- #
# PhysioNet BIG IDEAs (pre-reduced by scripts/fetch_bigideas.py)
# --------------------------------------------------------------------------- #
def load_bigideas(dir_path: str = BIGIDEAS_DIR, subjects: list[str] | None = None,
                  lookback_s: float = 60.0, win: int = WIN, max_windows_per_egv: int = 12,
                  band: tuple[float, float] | None = None, keep_amplitude: bool = False) -> WindowSet:
    """5-s windows from the ``lookback_s`` seconds preceding each CGM reading.

    The npz stores 120-s BVP windows centred on the CGM time (60 s before,
    60 s after).  Only the pre-reading part is used so the model never sees
    the future, and windows containing any gap (NaN) are dropped.
    """
    files = sorted(glob.glob(os.path.join(dir_path, "sub_*.npz")))
    if subjects:
        files = [f for f in files if os.path.basename(f)[4:7] in subjects]
    if not files:
        raise FileNotFoundError(f"no sub_*.npz under {dir_path}")
    Xs, ys, subj, rec, meta = [], [], [], [], []
    for f in files:
        sid = os.path.basename(f)[4:7]
        d = np.load(f)
        bvp, glucose, t = d["bvp"], d["glucose"], d["egv_time"]
        pre = int(round(bvp.shape[1] / 2))                  # index of the CGM instant
        start = pre - int(lookback_s * FS)
        kept = 0
        for i in range(len(glucose)):
            if not (40 <= glucose[i] <= 400):
                continue
            seg = bvp[i, start:pre]
            if band is not None:                              # optional re-filtering of the 64-Hz BVP
                ok = np.isfinite(seg)
                if ok.sum() < win:
                    continue
                seg = np.where(ok, seg, np.nan)
                filled = np.where(ok, seg, np.nanmean(seg))
                seg = np.where(ok, bandpass(filled, FS, band[0], band[1]), np.nan)
            W = sliding_windows(seg, win, win)
            W = W[np.isfinite(W).all(1) & (W.std(1) > 1e-3)][-max_windows_per_egv:]   # nearest the reading
            if not len(W):
                continue
            key = f"{sid}_{i:05d}"
            Xs.append(W); ys.append(np.full(len(W), glucose[i], np.float32))
            subj.append(np.full(len(W), sid)); rec.append(np.full(len(W), key))
            meta.append({"record": key, "subject": sid, "glucose": float(glucose[i]),
                         "egv_time": float(t[i]), "n_windows": int(len(W)), "raw_std": float(W.std(1).mean())})
            kept += 1
        log.info("BIG IDEAs sub_%s: %d/%d CGM readings have gap-free BVP", sid, kept, len(glucose))
    X = np.concatenate(Xs)
    if keep_amplitude:                                        # per-subject scale from the WHOLE subject (no labels used)
        subj_all = np.concatenate(subj)
        X = X.astype(np.float32).copy()
        for s in np.unique(subj_all):
            m = subj_all == s
            X[m] = (X[m] - X[m].mean(1, keepdims=True)) / (np.median(X[m].std(1)) + 1e-8)
    else:
        X = _zscore_rows(X)
    ws = WindowSet(X, np.concatenate(ys), np.concatenate(subj),
                   np.concatenate(rec), pd.DataFrame(meta), source="BIG IDEAs")
    log.info(ws.describe())
    return ws


# --------------------------------------------------------------------------- #
# STUBA ePPG (Berki & Mateasik, Zenodo 21978226, CC0): 24 subjects, 125 records
# --------------------------------------------------------------------------- #
STUBA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.nosync", "stuba", "wfdb_data")
STUBA_FS = 100
MMOL_TO_MGDL = 18.016


def load_stuba(root: str = STUBA_ROOT, channel: str = "IR1", win: int = WIN, hop: int = HOP) -> WindowSet:
    """30-s, 100-Hz finger ePPG records (channels R1, IR1, G1, R2, IR2, G2 =
    655/940/530 nm at two sites, plus accelerometer X/Y/Z); one capillary
    reference per record stored as a WFDB annotation 'GLY <mmol/L>'.
    ``channel`` may also be 'mean' (average of the six PPG channels)."""
    import wfdb
    subjects = pd.read_csv(os.path.join(root, "subjects.csv")).set_index("name")
    records = [l.strip().replace("\\", "/") for l in open(os.path.join(root, "RECORDS")) if l.strip()]
    Xs, ys, subj, rec, meta = [], [], [], [], []
    for r in records:
        sig = wfdb.rdrecord(os.path.join(root, r))
        ann = wfdb.rdann(os.path.join(root, r), "atr")
        gly = [a for a in ann.aux_note if a.startswith("GLY")]
        if not gly:
            log.warning("%s: no glucose annotation; skipped", r)
            continue
        glucose = float(gly[0].split()[1]) * MMOL_TO_MGDL
        names = sig.sig_name
        if channel == "mean":
            raw = sig.p_signal[:, [names.index(c) for c in ("R1", "IR1", "G1", "R2", "IR2", "G2")]].mean(1)
        else:
            raw = sig.p_signal[:, names.index(channel)]
        x = to_common_rate(raw.astype(np.float64), STUBA_FS)
        W = sliding_windows(x, win, hop)
        sname = r.split("/")[0]
        info = subjects.loc[sname]
        key = f"{int(info['id']):02d}_{r.split('/')[-1][-4:]}"
        Xs.append(W); ys.append(np.full(len(W), glucose, np.float32))
        subj.append(np.full(len(W), f"{int(info['id']):02d}")); rec.append(np.full(len(W), key))
        meta.append({"record": key, "subject": f"{int(info['id']):02d}", "glucose": glucose,
                     "gender": "Male" if info["sex"] == "M" else "Female", "age": float(info["age"]),
                     "height": float(info["height"]), "weight": float(info["weight"]),
                     "diabetes": str(info["diabetes"]), "channel": channel, "raw_std": float(np.std(raw)),
                     "n_windows": int(len(W))})
    ws = WindowSet(_zscore_rows(np.concatenate(Xs)), np.concatenate(ys), np.concatenate(subj),
                   np.concatenate(rec), pd.DataFrame(meta), source=f"STUBA ePPG ({channel})")
    log.info(ws.describe())
    return ws


# --------------------------------------------------------------------------- #
# Model views (same transforms as the original pipeline, computed once)
# --------------------------------------------------------------------------- #
def cwt_scalogram(x: np.ndarray) -> np.ndarray:
    coef, _ = pywt.cwt(x.astype(np.float64), np.arange(1, CWT_SCALES + 1, dtype=np.float64), "mexh")
    mag = cv2.resize(np.abs(coef).astype(np.float32), (SPEC, SPEC), interpolation=cv2.INTER_LINEAR)
    mn, mx = mag.min(), mag.max()
    return (mag - mn) / (mx - mn) if mx - mn > 1e-8 else mag


def phase_space(x: np.ndarray, tau: int = TAU, n_points: int = PHASE_POINTS) -> np.ndarray:
    L = len(x) - 2 * tau
    pts = np.vstack((x[:L], x[tau:tau + L], x[2 * tau:2 * tau + L])).astype(np.float32)
    if pts.shape[1] >= n_points:                       # spread points over the whole window
        idx = np.linspace(0, pts.shape[1] - 1, n_points).round().astype(int)
        return pts[:, idx]
    return np.pad(pts, ((0, 0), (0, n_points - pts.shape[1])))


def make_views(X: np.ndarray, which: tuple[str, ...] = ("x1", "x2", "x3")) -> dict[str, torch.Tensor]:
    """Return {'x1': (N,1,L), 'x2': (N,1,64,64), 'x3': (N,3,256)} float32 tensors
    (only the requested views are computed; the others are 1-element dummies)."""
    out = {"x1": torch.from_numpy(X[:, None, :].astype(np.float32))}
    out["x2"] = (torch.from_numpy(np.stack([cwt_scalogram(x) for x in X])[:, None].astype(np.float32))
                 if "x2" in which else torch.zeros(len(X), 1, 1, 1))
    out["x3"] = (torch.from_numpy(np.stack([phase_space(x) for x in X]).astype(np.float32))
                 if "x3" in which else torch.zeros(len(X), 3, 1))
    return out


# --------------------------------------------------------------------------- #
# Hand-crafted features for the classical baseline
# --------------------------------------------------------------------------- #
def handcrafted_features(X: np.ndarray) -> np.ndarray:
    """Per-window PPG descriptors: HR, spectral bands, morphology moments."""
    from scipy.stats import kurtosis, skew
    f, P = sps.welch(X, fs=FS, nperseg=min(256, X.shape[1]), axis=1)
    band = lambda lo, hi: P[:, (f >= lo) & (f < hi)].sum(1)
    tot = P.sum(1) + 1e-12
    hr_band = (f >= 0.7) & (f <= 3.0)
    hr = f[hr_band][P[:, hr_band].argmax(1)] * 60.0
    feats = np.column_stack([
        hr, band(0.5, 1.5) / tot, band(1.5, 3.0) / tot, band(3.0, 8.0) / tot,
        np.log(tot), skew(X, axis=1), kurtosis(X, axis=1),
        np.abs(np.diff(X, axis=1)).mean(1), (np.diff(np.sign(np.diff(X, axis=1)), axis=1) != 0).mean(1),
    ])
    return np.nan_to_num(feats).astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ws = load_mendeley()
    v = make_views(ws.X[:4])
    print({k: tuple(t.shape) for k, t in v.items()})
    print(ws.meta.groupby("subject").glucose.agg(["size", "min", "max"]).T.to_string())
    print(json.dumps({"windows": len(ws), "features": handcrafted_features(ws.X).shape}, indent=1))
