"""
ppgbench.data.bigideas
======================
Loader for the PhysioNet "BIG IDEAs Lab Glycemic Variability and Wearable
Device" dataset (v1.1.3): Empatica E4 wrist BVP at 64 Hz against Dexcom G6
CGM readings every 5 minutes, 16 subjects over ~10 days each.  It is the only
open dataset with genuine glycaemic *dynamics* inside each subject, which is
what makes chronological (within-subject) protocols meaningful.

Pre-reduced files
-----------------
The raw BVP CSVs are ~1.4 GB per subject, so ``scripts/fetch_bigideas.py``
(wrapped by :func:`fetch_bigideas`) streams them once into
``<dir>/sub_XXX.npz`` holding, per CGM reading, a 120-s BVP excerpt centred on
the reading time (``bvp (n, 7680)`` with NaN where the device was off),
``glucose (n,)``, ``egv_time (n,)`` POSIX seconds and ``roc (n,)``.

What this loader does (identical to ``ppg_data.load_bigideas``)
----------------------------------------------------------------
For each CGM reading with 40 <= glucose <= 400 mg/dL it takes the
``lookback_s`` seconds *preceding* the reading (never the future), cuts
non-overlapping 5-s windows, drops windows with any gap (NaN) or a flat
signal, keeps the ``max_windows_per_egv`` windows nearest the reading, and
z-scores each.  One CGM reading is one ``record``; ``time`` is its
``egv_time``, which the chronological protocols split on.

Note that the pre-reduced BVP is already at 64 Hz but *not* band-passed; the
legacy pipeline (and therefore this loader) relies on per-window z-scoring
only, so any change here would break numeric parity.
"""
from __future__ import annotations

import glob
import importlib.util
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppgbench.core.signal import sliding_windows, zscore_rows
from ppgbench.core.windowset import WindowSet
from ppgbench.data.registry import FS, WIN, DatasetError, register_dataset

log = logging.getLogger("ppgbench.data.bigideas")

__all__ = ["GLUCOSE_RANGE", "NPZ_KEYS", "default_dir", "is_available", "list_subjects", "normalise_subjects",
           "read_subject", "egv_windows", "load_bigideas", "fetch_bigideas"]

GLUCOSE_RANGE: tuple[float, float] = (40.0, 400.0)   #: Dexcom G6 reportable range (inclusive)
NPZ_KEYS: tuple[str, ...] = ("egv_time", "glucose", "bvp")  #: required arrays in sub_XXX.npz
MIN_WINDOW_STD: float = 1e-3                           #: below this the window is a flat line (device off)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR = _REPO_ROOT / "data.nosync" / "bigideas"
_FETCH_SCRIPT = _REPO_ROOT / "scripts" / "fetch_bigideas.py"


def default_dir() -> str:
    """Directory of ``sub_XXX.npz`` files: ``$PPGBENCH_BIGIDEAS`` if set, else ``data.nosync/bigideas``."""
    return os.environ.get("PPGBENCH_BIGIDEAS", str(_DEFAULT_DIR))


def is_available(dir_path: str | None = None) -> bool:
    """``True`` when at least one ``sub_*.npz`` exists."""
    return bool(glob.glob(os.path.join(dir_path or default_dir(), "sub_*.npz")))


def normalise_subjects(subjects: Iterable[str | int] | None) -> list[str] | None:
    """Accept ``["001", 1, "1"]`` and return zero-padded three-character ids."""
    if subjects is None:
        return None
    out = []
    for s in subjects:
        s = str(s).strip()
        out.append(f"{int(s):03d}" if s.isdigit() else s)
    return out


def list_subjects(dir_path: str | None = None) -> list[str]:
    """Sorted subject ids with a ``sub_XXX.npz`` under ``dir_path``."""
    files = sorted(glob.glob(os.path.join(dir_path or default_dir(), "sub_*.npz")))
    return [os.path.basename(f)[4:7] for f in files]


def read_subject(path: str) -> dict[str, np.ndarray]:
    """Load and sanity-check one ``sub_XXX.npz``.

    Returns a dict with ``egv_time (n,) float64``, ``glucose (n,) float32``,
    ``bvp (n, L) float32`` and, when present, ``roc`` and ``bvp_coverage``.
    """
    d = np.load(path)
    missing = [k for k in NPZ_KEYS if k not in d.files]
    if missing:
        raise DatasetError(f"{path}: missing arrays {missing}")
    out = {"egv_time": d["egv_time"].astype(np.float64), "glucose": d["glucose"], "bvp": d["bvp"]}
    for k in ("roc", "bvp_coverage"):
        if k in d.files:
            out[k] = d[k]
    n = len(out["glucose"])
    if len(out["egv_time"]) != n or out["bvp"].ndim != 2 or out["bvp"].shape[0] != n:
        raise DatasetError(f"{path}: inconsistent shapes egv_time {out['egv_time'].shape} glucose "
                           f"{out['glucose'].shape} bvp {out['bvp'].shape}")
    if n and np.any(np.diff(out["egv_time"]) < 0):
        log.warning("%s: egv_time is not sorted", path)
    return out


def egv_windows(bvp_row: np.ndarray, lookback_s: float = 60.0, win: int = WIN, fs: int = FS,
                max_windows: int = 12) -> np.ndarray:
    """Gap-free, non-flat, non-overlapping windows from the ``lookback_s`` s before the CGM instant.

    The excerpt is centred on the reading, so the CGM instant is at index
    ``round(L / 2)``; only samples before it are used so no model ever sees
    the future.  The last ``max_windows`` windows (nearest the reading) are
    returned, possibly zero rows.
    """
    pre = int(round(len(bvp_row) / 2))
    start = pre - int(lookback_s * fs)
    W = sliding_windows(bvp_row[max(start, 0):pre], win, win)
    if not len(W):
        return W
    return W[np.isfinite(W).all(1) & (W.std(1) > MIN_WINDOW_STD)][-max_windows:]


@register_dataset(
    "bigideas",
    description="PhysioNet BIG IDEAs (v1.1.3): Empatica E4 wrist BVP (64 Hz) vs Dexcom G6 CGM every 5 min, "
                "16 subjects x ~10 days; one record per CGM reading, timestamps available.",
    citation="Bent B. et al., 'Engineering digital biomarkers of interstitial glucose from noninvasive "
             "smartwatches', npj Digit. Med. 4, 89 (2021); PhysioNet doi:10.13026/zthx-5212.",
    license="PhysioNet Open Data (ODC-By 1.0)",
    url="https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/",
    availability=is_available,
    tags=("ppg", "bvp", "cgm", "wrist", "longitudinal"),
)
def load_bigideas(dir_path: str | None = None, subjects: Sequence[str | int] | None = None,
                  lookback_s: float = 60.0, win: int = WIN, max_windows_per_egv: int = 12,
                  glucose_range: tuple[float, float] = GLUCOSE_RANGE) -> WindowSet:
    """5-s windows preceding each CGM reading, one record per reading.

    Parameters
    ----------
    dir_path : directory of ``sub_XXX.npz`` (default :func:`default_dir`).
    subjects : ids to keep (``"001"`` or ``1``); ``None`` keeps every file.
    lookback_s : seconds before the reading from which windows are cut (<= 60).
    win : window length in samples at 64 Hz.
    max_windows_per_egv : keep at most this many windows nearest each reading
        (the paper uses 2 for the within-subject runs, 12 at most).
    glucose_range : inclusive reportable range; readings outside are dropped.

    Returns
    -------
    WindowSet with ``time = egv_time`` per window and ``meta`` columns
    ``record, subject, glucose, egv_time, roc, bvp_coverage, n_windows``.

    Numerically identical to ``ppg_data.load_bigideas`` for the default
    ``glucose_range``.
    """
    dir_path = dir_path or default_dir()
    files = sorted(glob.glob(os.path.join(dir_path, "sub_*.npz")))
    keep = normalise_subjects(subjects)
    if keep:
        files = [f for f in files if os.path.basename(f)[4:7] in keep]
    if not files:
        raise FileNotFoundError(f"no sub_*.npz under {dir_path} (subjects={subjects}); run fetch_bigideas() "
                                f"or set PPGBENCH_BIGIDEAS")
    lo, hi = glucose_range
    Xs, ys, subj, rec, times, meta = [], [], [], [], [], []
    for f in files:
        sid = os.path.basename(f)[4:7]
        d = read_subject(f)
        bvp, glucose, t = d["bvp"], d["glucose"], d["egv_time"]
        roc = d.get("roc")
        cov = d.get("bvp_coverage")
        kept = 0
        for i in range(len(glucose)):
            if not (lo <= glucose[i] <= hi):
                continue
            W = egv_windows(bvp[i], lookback_s, win, FS, max_windows_per_egv)
            if not len(W):
                continue
            key = f"{sid}_{i:05d}"
            Xs.append(W)
            ys.append(np.full(len(W), glucose[i], np.float32))
            subj.append(np.full(len(W), sid))
            rec.append(np.full(len(W), key))
            times.append(np.full(len(W), float(t[i]), np.float64))
            meta.append({"record": key, "subject": sid, "glucose": float(glucose[i]),
                         "egv_time": float(t[i]),
                         "roc": float(roc[i]) if roc is not None else np.nan,
                         "bvp_coverage": float(cov[i]) if cov is not None else np.nan,
                         "n_windows": int(len(W))})
            kept += 1
        log.info("BIG IDEAs sub_%s: %d/%d CGM readings have gap-free BVP", sid, kept, len(glucose))
    if not Xs:
        raise DatasetError(f"no CGM reading under {dir_path} has usable BVP (subjects={subjects})")
    ws = WindowSet(X=zscore_rows(np.concatenate(Xs)), y=np.concatenate(ys),
                   subject=np.concatenate(subj), record=np.concatenate(rec),
                   time=np.concatenate(times), meta=pd.DataFrame(meta), fs=FS, source="BIG IDEAs")
    log.info(ws.describe())
    return ws


# --------------------------------------------------------------------------- #
# Fetch wrapper around scripts/fetch_bigideas.py
# --------------------------------------------------------------------------- #
def _load_fetch_module(script: Path | None = None) -> Any:
    script = Path(script) if script else _FETCH_SCRIPT
    if not script.exists():
        raise FileNotFoundError(f"fetch script not found: {script}")
    spec = importlib.util.spec_from_file_location("ppgbench_fetch_bigideas", script)
    if spec is None or spec.loader is None:   # pragma: no cover
        raise ImportError(f"cannot import {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_bigideas(out_dir: str | None = None, *, zip_path: str | None = None, dir_path: str | None = None,
                   subjects: Sequence[str | int] | None = None, signals: Sequence[str] = ("BVP",),
                   force: bool = False, script: str | None = None) -> list[dict[str, Any]]:
    """Reduce the raw PhysioNet archive into ``sub_XXX.npz`` files.

    Thin wrapper over ``scripts/fetch_bigideas.py`` so the CLI and notebooks
    can prepare the data without shelling out.  Exactly one of ``zip_path``
    (the PhysioNet zip) or ``dir_path`` (an unpacked ``1.1.3/`` folder) must
    be given.  ``signals`` defaults to BVP only, which is all the benchmark
    needs and keeps each file to ~80 MB.

    Returns the per-subject summary dicts the script writes to
    ``sub_XXX_summary.json``.  Subjects whose files are missing are logged and
    skipped rather than aborting the whole run.
    """
    if (zip_path is None) == (dir_path is None):
        raise ValueError("fetch_bigideas: give exactly one of zip_path= or dir_path=")
    out = Path(out_dir or default_dir())
    out.mkdir(parents=True, exist_ok=True)
    mod = _load_fetch_module(Path(script) if script else None)
    src = mod.Source(Path(zip_path) if zip_path else None, Path(dir_path) if dir_path else None)
    ids = normalise_subjects(subjects) or [f"{i:03d}" for i in range(1, 17)]
    summaries: list[dict[str, Any]] = []
    for s in ids:
        try:
            summaries.append(mod.process_subject(src, s, out, force, tuple(signals)))
        except FileNotFoundError as exc:
            log.error("sub_%s: missing %s", s, exc)
    return summaries
