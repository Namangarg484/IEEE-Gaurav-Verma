"""
ppgbench.data.mendeley
======================
Loader for the Mendeley "PPG dataset for blood glucose" (spot measurements).

The dataset
-----------
67 fingertip PPG recordings from 23 subjects.  Each recording is a single
column of 21,900 samples (10 s at 2175 Hz) stored as
``RawData/signal_SS_RRRR.csv`` and labelled by ``Labels/label_SS_RRRR.csv``
with ONE reference capillary glucose value plus age, sex, height and weight.
``SS`` is the subject id and ``RRRR`` the recording index, so the subject is
the split key for subject-wise protocols and the recording is the unit of
independent reference measurements (n = 67).

What this loader does (identical to ``ppg_data.load_mendeley``)
----------------------------------------------------------------
1. remove the DC offset, band-pass 0.5-8 Hz at the native 2175 Hz and
   polyphase-resample to the common 64 Hz (``ppgbench.core.signal.to_rate``);
2. cut 5-s windows (320 samples) with 50 % overlap -- the recordings are only
   10 s long, so overlap is the only way to obtain more than two windows;
3. z-score every window so amplitude (which depends on contact pressure and
   skin, not glucose) carries no information.

The resulting ``WindowSet`` has ``time=None``: the recordings carry no
timestamps, so chronological protocols are undefined for this source.
``meta`` holds one row per recording with the demographics the ``demo``
baseline uses (``age``, ``sex``, ``height``, ``weight``; ``gender`` is kept as
an alias of ``sex`` for the legacy scripts).

Important caveat (why this benchmark exists)
--------------------------------------------
``Archive_Dataset/clean-dataset.csv`` that ships next to the raw files is a
cross-join of every PPG sample with every label of the same subject and must
not be used.  This loader reads the raw recordings only.
"""
from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ppgbench.core.signal import sliding_windows, to_rate, zscore_rows
from ppgbench.core.windowset import WindowSet
from ppgbench.data.registry import FS, HOP, WIN, DatasetError, register_dataset

log = logging.getLogger("ppgbench.data.mendeley")

__all__ = ["MENDELEY_FS", "RECORD_SAMPLES", "default_root", "is_available", "list_records",
           "read_record", "record_windows", "load_mendeley"]

MENDELEY_FS: int = 2175        #: native sampling rate of the pulse sensor (Hz)
RECORD_SAMPLES: int = 21_900   #: samples per recording (10 s)
_SIG_PREFIX, _SIG_SUFFIX = "signal_", ".csv"
_LABEL_COLS = ("Glucose", "Gender", "Age", "Height", "Weight")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "Archive_Dataset" / "archive (2)" / "PPG_Dataset"


def default_root() -> str:
    """Raw-data directory: ``$PPGBENCH_MENDELEY`` if set, else the in-repo location."""
    return os.environ.get("PPGBENCH_MENDELEY", str(_DEFAULT_ROOT))


def is_available(root: str | None = None) -> bool:
    """``True`` when at least one ``RawData/signal_*.csv`` exists under ``root``."""
    root = root or default_root()
    return bool(glob.glob(os.path.join(root, "RawData", f"{_SIG_PREFIX}*{_SIG_SUFFIX}")))


def list_records(root: str | None = None) -> list[str]:
    """Sorted ``SS_RRRR`` keys that have both a signal and a label file."""
    root = root or default_root()
    keys = []
    for sf in sorted(glob.glob(os.path.join(root, "RawData", f"{_SIG_PREFIX}*{_SIG_SUFFIX}"))):
        key = os.path.basename(sf)[len(_SIG_PREFIX):-len(_SIG_SUFFIX)]
        if os.path.exists(os.path.join(root, "Labels", f"label_{key}.csv")):
            keys.append(key)
    return keys


def read_record(root: str, key: str) -> tuple[np.ndarray, dict[str, float | str]]:
    """Return ``(raw float64 samples at 2175 Hz, label dict)`` for one recording.

    The label dict has keys ``glucose, sex, age, height, weight`` (floats and
    the sex string exactly as in the CSV).
    """
    sf = os.path.join(root, "RawData", f"{_SIG_PREFIX}{key}{_SIG_SUFFIX}")
    lf = os.path.join(root, "Labels", f"label_{key}.csv")
    raw = pd.read_csv(sf, header=None).to_numpy(np.float64).ravel()
    lab = pd.read_csv(lf)
    missing = [c for c in _LABEL_COLS if c not in lab.columns]
    if missing:
        raise DatasetError(f"{lf}: missing label columns {missing}")
    row = lab.iloc[0]
    return raw, {"glucose": float(row["Glucose"]), "sex": str(row["Gender"]), "age": float(row["Age"]),
                 "height": float(row["Height"]), "weight": float(row["Weight"])}


def record_windows(raw: np.ndarray, win: int = WIN, hop: int = HOP, fs_in: int = MENDELEY_FS) -> np.ndarray:
    """Band-pass, resample to 64 Hz and cut overlapping windows (not yet z-scored)."""
    x = to_rate(raw, fs_in, FS)
    return sliding_windows(x, win, hop)


def _subject_of(key: str) -> str:
    return key.split("_")[0]


@register_dataset(
    "mendeley",
    description="Mendeley PPG-glucose spot measurements: 67 x 10-s fingertip recordings (2175 Hz) from 23 "
                "subjects, one capillary glucose reference per recording; no timestamps.",
    citation="A. Kermani and H. Esmaeili, 'The dataset of photoplethysmography signals collected from a pulse "
             "sensor to measure blood glucose level', Mendeley Data v3, 2023 (doi:10.17632/37pm7jk7jn.3).",
    license="CC BY 4.0",
    url="https://data.mendeley.com/datasets/37pm7jk7jn",
    availability=is_available,
    tags=("ppg", "spot", "fingertip"),
)
def load_mendeley(root: str | None = None, win: int = WIN, hop: int = HOP,
                  subjects: Iterable[str] | None = None) -> WindowSet:
    """Load the raw Mendeley recordings as z-scored 64-Hz windows.

    Parameters
    ----------
    root : dataset directory containing ``RawData/`` and ``Labels/`` (default
        :func:`default_root`).
    win, hop : window length and hop in samples at 64 Hz (320 / 160 = 5 s, 50 %).
    subjects : optional subject ids (``"01"`` ...) to keep; ``None`` keeps all.

    Returns
    -------
    WindowSet with ``time=None`` and ``meta`` columns ``record, subject, glucose,
    sex, gender, age, height, weight, raw_std, n_windows``.

    Numerically identical to ``ppg_data.load_mendeley`` (same filtering,
    resampling, windowing, ordering and per-window z-scoring).
    """
    root = root or default_root()
    sig_files = sorted(glob.glob(os.path.join(root, "RawData", f"{_SIG_PREFIX}*{_SIG_SUFFIX}")))
    if not sig_files:
        raise FileNotFoundError(f"no {_SIG_PREFIX}*{_SIG_SUFFIX} under {os.path.join(root, 'RawData')}; "
                                f"set PPGBENCH_MENDELEY or pass root=")
    keep = None if subjects is None else {str(s) for s in subjects}
    Xs, ys, subj, rec, meta = [], [], [], [], []
    for sf in sig_files:
        key = os.path.basename(sf)[len(_SIG_PREFIX):-len(_SIG_SUFFIX)]
        sid = _subject_of(key)
        if keep is not None and sid not in keep:
            continue
        if not os.path.exists(os.path.join(root, "Labels", f"label_{key}.csv")):
            log.warning("no label for %s; skipped", key)
            continue
        raw, lab = read_record(root, key)
        if len(raw) != RECORD_SAMPLES:
            log.warning("%s has %d samples (expected %d)", key, len(raw), RECORD_SAMPLES)
        W = record_windows(raw, win, hop)
        if not len(W):
            log.warning("%s too short for a single window; skipped", key)
            continue
        Xs.append(W)
        ys.append(np.full(len(W), lab["glucose"], np.float32))
        subj.append(np.full(len(W), sid))
        rec.append(np.full(len(W), key))
        meta.append({"record": key, "subject": sid, "glucose": lab["glucose"],
                     "sex": lab["sex"], "gender": lab["sex"], "age": lab["age"],
                     "height": lab["height"], "weight": lab["weight"],
                     "raw_std": float(raw.std()), "n_windows": int(len(W))})
    if not Xs:
        raise DatasetError(f"no usable recordings under {root} (subjects={subjects})")
    ws = WindowSet(X=zscore_rows(np.concatenate(Xs)), y=np.concatenate(ys),
                   subject=np.concatenate(subj), record=np.concatenate(rec), time=None,
                   meta=pd.DataFrame(meta), fs=FS, source="Mendeley")
    log.info(ws.describe())
    return ws
