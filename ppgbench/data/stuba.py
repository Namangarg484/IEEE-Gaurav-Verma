"""
ppgbench.data.stuba
===================
Loader for the STUBA ePPG multi-wavelength finger-PPG dataset (Berki &
Mateasik, Zenodo 10.5281/zenodo.21978226, CC0).

The dataset
-----------
125 WFDB records (30 s at 100 Hz) from 24 subjects, 3 of them with type-1
diabetes; each record carries one capillary reference (Beurer GL49) as a
WFDB annotation ``GLY <mmol/L>``.  Channels: ``R1, IR1, G1, R2, IR2, G2``
(655 / 940 / 530 nm at two sites) plus accelerometer ``X, Y, Z``.  Subject
metadata (age, sex, height, weight, diabetes) is in ``subjects.csv``.

What this loader does (identical to ``ppg_data.load_stuba``)
-------------------------------------------------------------
Selects one PPG channel (default IR1, 940 nm) or the mean of the six, removes
the DC offset, band-passes 0.5-8 Hz at 100 Hz, resamples to 64 Hz, cuts 5-s
windows with 50 % overlap and z-scores each window.  ``time`` is None (no
timestamps); for the two diabetic subjects with 32 and 53 records the record
index is the only chronological order available.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ppgbench.core.signal import sliding_windows, to_rate, zscore_rows
from ppgbench.core.windowset import WindowSet
from ppgbench.data.registry import HOP, WIN, DatasetError, register_dataset

log = logging.getLogger("ppgbench.data.stuba")

__all__ = ["STUBA_FS", "MMOL_TO_MGDL", "PPG_CHANNELS", "default_root", "is_available", "load_stuba"]

STUBA_FS: int = 100
MMOL_TO_MGDL: float = 18.016
PPG_CHANNELS = ("R1", "IR1", "G1", "R2", "IR2", "G2")
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data.nosync" / "stuba" / "wfdb_data"


def default_root() -> str:
    return os.environ.get("PPGBENCH_STUBA", str(_DEFAULT_ROOT))


def is_available(root: str | None = None) -> bool:
    root = root or default_root()
    return os.path.exists(os.path.join(root, "RECORDS")) and os.path.exists(os.path.join(root, "subjects.csv"))


@register_dataset(
    "stuba",
    description="STUBA ePPG: 125 x 30-s six-channel finger PPG records (100 Hz) from 24 subjects (3 T1DM), one "
                "capillary glucose reference per record; no timestamps.",
    citation="M. Berki and A. Mateasik, 'Photoplethysmography (PPG) dataset for non-invasive blood glucose "
             "monitoring', Zenodo, 2026, doi:10.5281/zenodo.21978226.",
    license="CC0 1.0",
    url="https://zenodo.org/records/21978226",
    availability=is_available,
    tags=("ppg", "spot", "fingertip", "multi-wavelength", "diabetes"),
)
def load_stuba(root: str | None = None, channel: str = "IR1", win: int = WIN, hop: int = HOP,
               subjects: list[str] | None = None) -> WindowSet:
    """Load the ePPG records as z-scored 64-Hz windows of one channel (or ``'mean'``)."""
    import wfdb

    root = root or default_root()
    if not is_available(root):
        raise DatasetError(f"STUBA ePPG not found under {root} (set $PPGBENCH_STUBA or unzip stuba_ppg_bgl_dataset.zip)")
    if channel != "mean" and channel not in PPG_CHANNELS:
        raise DatasetError(f"channel must be one of {PPG_CHANNELS} or 'mean', got {channel!r}")
    subj_table = pd.read_csv(os.path.join(root, "subjects.csv")).set_index("name")
    records = [l.strip().replace("\\", "/") for l in open(os.path.join(root, "RECORDS")) if l.strip()]
    Xs, ys, subj, rec, meta = [], [], [], [], []
    for r in records:
        info = subj_table.loc[r.split("/")[0]]
        sid = f"{int(info['id']):02d}"
        if subjects is not None and sid not in subjects:
            continue
        sig = wfdb.rdrecord(os.path.join(root, r))
        ann = wfdb.rdann(os.path.join(root, r), "atr")
        gly = [a for a in ann.aux_note if a.startswith("GLY")]
        if not gly:
            log.warning("%s: no glucose annotation; skipped", r)
            continue
        glucose = float(gly[0].split()[1]) * MMOL_TO_MGDL
        names = sig.sig_name
        raw = (sig.p_signal[:, [names.index(c) for c in PPG_CHANNELS]].mean(1) if channel == "mean"
               else sig.p_signal[:, names.index(channel)]).astype(np.float64)
        x = to_rate(raw, STUBA_FS)
        W = sliding_windows(x, win, hop)
        key = f"{sid}_{r.split('/')[-1][-4:]}"
        Xs.append(W); ys.append(np.full(len(W), glucose, np.float32))
        subj.append(np.full(len(W), sid)); rec.append(np.full(len(W), key))
        sex = "Male" if str(info["sex"]).upper().startswith("M") else "Female"
        meta.append({"record": key, "subject": sid, "glucose": glucose, "sex": sex, "gender": sex,
                     "age": float(info["age"]), "height": float(info["height"]), "weight": float(info["weight"]),
                     "diabetes": str(info["diabetes"]), "channel": channel, "raw_std": float(np.std(raw)),
                     "n_windows": int(len(W))})
    if not Xs:
        raise DatasetError("no STUBA records loaded")
    return WindowSet(X=zscore_rows(np.concatenate(Xs)), y=np.concatenate(ys), subject=np.concatenate(subj),
                     record=np.concatenate(rec), time=None, meta=pd.DataFrame(meta), fs=64,
                     source=f"STUBA ePPG ({channel})")
