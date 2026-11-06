"""
ppgbench.core.windowset
=======================
The :class:`WindowSet` container: fixed-length, per-window z-scored signal
windows together with the three labels every leakage-controlled evaluation
needs.

Why three keys per window?
--------------------------
A PPG-glucose dataset has a hierarchy *subject > record > window*:

* ``subject`` - the person.  Identity is strongly encoded in PPG morphology,
  so any split that shares a subject between train and test lets a model
  regress the *person* rather than the glucose.  Subject-wise protocols group
  on this key.
* ``record``  - one independent reference measurement (a finger-prick or one
  CGM reading).  Overlapping windows of the same record carry the *same*
  label, so they are not independent samples; metrics are reported after
  averaging predictions per record (:func:`ppgbench.metrics.record_level`).
* ``time``    - POSIX seconds of the reference measurement, used by the
  chronological protocols and by the "time" level of the no-leak check.  It
  is ``None`` for datasets without timestamps (Mendeley).

``meta`` holds one row per record (``record``, ``subject``, ``glucose`` and
optional demographics / ``egv_time``) and is the single source of truth for
record-level attributes; per-window arrays are derived from it.
"""
from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("ppgbench.core.windowset")

__all__ = ["WindowSet", "concat_windowsets"]

#: Column names accepted for the sex / gender attribute, in lookup order.
_SEX_ALIASES: tuple[str, ...] = ("sex", "gender")
#: Values (lower-cased) that encode "male" -> 1.0; everything else -> 0.0.
#: Reproduces ``run_subject_cv.py`` (``meta["gender"] == "Male"``).
_MALE_TOKENS: frozenset[str] = frozenset({"male", "m", "1", "1.0", "true"})


@dataclass
class WindowSet:
    """Windows plus the labels / groups needed for subject-wise evaluation.

    Attributes
    ----------
    X : (N, L) float32
        Signal windows, z-scored per window (:func:`ppgbench.core.signal.zscore_rows`).
    y : (N,) float32
        Reference target (blood glucose, mg/dL) of the record each window belongs to.
    subject : (N,) str
        Split key for subject-wise protocols.
    record : (N,) str
        Identifier of the independent reference measurement.
    time : (N,) float64 or None
        POSIX seconds of the reference measurement, ``None`` if unknown.
    meta : DataFrame
        One row per record: ``record``, ``subject``, ``glucose`` and optional
        ``age``, ``sex``/``gender``, ``height``, ``weight``, ``egv_time`` ...
    fs : int
        Sampling rate of ``X`` in Hz (64 for every bundled loader).
    source : str
        Human-readable dataset name used in logs and reports.
    """

    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    record: np.ndarray
    time: np.ndarray | None = None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    fs: int = 64
    source: str = ""

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(np.asarray(self.X, dtype=np.float32))
        self.y = np.asarray(self.y, dtype=np.float32).reshape(-1)
        self.subject = np.asarray(self.subject, dtype=str).reshape(-1)
        self.record = np.asarray(self.record, dtype=str).reshape(-1)
        if self.time is not None:
            self.time = np.asarray(self.time, dtype=np.float64).reshape(-1)
        if self.meta is None:
            self.meta = pd.DataFrame()
        elif not isinstance(self.meta, pd.DataFrame):
            self.meta = pd.DataFrame(self.meta)
        self.fs = int(self.fs)
        self.source = str(self.source)

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, L); got shape {self.X.shape}")
        n = self.X.shape[0]
        for name in ("y", "subject", "record"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows but X has {n}")
        if self.time is not None and self.time.shape[0] != n:
            raise ValueError(f"time has {self.time.shape[0]} rows but X has {n}")
        if n and not np.isfinite(self.X).all():
            raise ValueError("X contains NaN/inf; drop or impute windows before building a WindowSet")
        if len(self.meta) and "record" in self.meta.columns:
            if self.meta["record"].duplicated().any():
                dup = self.meta.loc[self.meta["record"].duplicated(), "record"].tolist()[:5]
                raise ValueError(f"meta must have one row per record; duplicated: {dup}")
            missing = set(np.unique(self.record)) - set(self.meta["record"].astype(str))
            if missing:
                raise ValueError(f"{len(missing)} record(s) in `record` have no meta row, e.g. {sorted(missing)[:5]}")

    # ------------------------------------------------------------------ #
    # basic protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"WindowSet({self.describe()})"

    @property
    def n_windows(self) -> int:
        return len(self)

    @property
    def n_records(self) -> int:
        return int(len(np.unique(self.record)))

    @property
    def n_subjects(self) -> int:
        return int(len(np.unique(self.subject)))

    @property
    def window_len(self) -> int:
        """Window length in samples."""
        return int(self.X.shape[1])

    @property
    def has_time(self) -> bool:
        return self.time is not None

    def describe(self) -> str:
        """One-line summary (same format as the legacy ``ppg_data.WindowSet``)."""
        if len(self) == 0:
            return f"{self.source}: 0 windows"
        return (f"{self.source}: {len(self)} windows, {self.n_records} records, "
                f"{self.n_subjects} subjects, glucose {self.y.min():.0f}-{self.y.max():.0f} "
                f"(mean {self.y.mean():.1f} +/- {self.y.std():.1f}) mg/dL")

    # ------------------------------------------------------------------ #
    # indexing
    # ------------------------------------------------------------------ #
    def subset(self, mask_or_idx: np.ndarray | Sequence[int] | Sequence[bool]) -> "WindowSet":
        """Return a new :class:`WindowSet` restricted to the given windows.

        ``mask_or_idx`` is either a boolean mask of length N or an integer
        index array (order and repeats are preserved, like numpy fancy
        indexing).  ``meta`` is restricted to the records that survive, in its
        original row order.
        """
        sel = np.asarray(mask_or_idx)
        if sel.dtype == bool:
            if sel.shape[0] != len(self):
                raise ValueError(f"boolean mask has length {sel.shape[0]}, expected {len(self)}")
            idx = np.flatnonzero(sel)
        else:
            idx = sel.astype(np.intp).reshape(-1)
        record = self.record[idx]
        if len(self.meta) and "record" in self.meta.columns:
            keep = self.meta["record"].astype(str).isin(set(record.tolist()))
            meta = self.meta.loc[keep].reset_index(drop=True)
        else:
            meta = self.meta.copy()
        return WindowSet(
            X=self.X[idx], y=self.y[idx], subject=self.subject[idx], record=record,
            time=None if self.time is None else self.time[idx],
            meta=meta, fs=self.fs, source=self.source,
        )

    def __getitem__(self, item: Any) -> "WindowSet":
        if isinstance(item, slice):
            item = np.arange(len(self))[item]
        return self.subset(np.asarray(item))

    def record_rows(self) -> dict[str, np.ndarray]:
        """Map ``record -> int array of window rows`` (first-appearance order)."""
        order = pd.unique(self.record)
        groups = pd.Series(np.arange(len(self))).groupby(self.record, sort=False).apply(np.asarray)
        return {str(r): np.asarray(groups[r], dtype=np.intp) for r in order}

    def subject_rows(self) -> dict[str, np.ndarray]:
        """Map ``subject -> int array of window rows`` (first-appearance order)."""
        order = pd.unique(self.subject)
        groups = pd.Series(np.arange(len(self))).groupby(self.subject, sort=False).apply(np.asarray)
        return {str(s): np.asarray(groups[s], dtype=np.intp) for s in order}

    # ------------------------------------------------------------------ #
    # record-level attributes broadcast to windows
    # ------------------------------------------------------------------ #
    def meta_per_window(self, cols: str | Iterable[str]) -> pd.DataFrame:
        """Broadcast record-level ``meta`` columns to one row per window."""
        if isinstance(cols, str):
            cols = [cols]
        cols = list(cols)
        if "record" not in self.meta.columns:
            raise KeyError("meta has no 'record' column")
        m = self.meta.assign(record=self.meta["record"].astype(str)).set_index("record")
        missing = [c for c in cols if c not in m.columns]
        if missing:
            raise KeyError(f"meta has no column(s) {missing}; available: {list(m.columns)}")
        return m.loc[self.record, cols].reset_index(drop=True)

    def demographics(self, cols: Sequence[str] = ("age", "height", "weight"),
                     sex_col: str = "sex") -> np.ndarray | None:
        """Per-window demographic matrix ``(N, k)`` float32, or ``None``.

        Reproduces the ``demo`` matrix of ``run_subject_cv.py``: the numeric
        columns in ``cols`` followed by a male indicator (1.0 for
        male/"M"/1, else 0.0) when a sex column exists.  ``sex_col`` is tried
        first, then the aliases ``("sex", "gender")``.  Returns ``None`` when
        any requested numeric column is missing from ``meta`` (e.g. BIG IDEAs
        has no demographics), so callers can use ``feats["demo"] is None`` as
        the capability test.
        """
        if not len(self.meta) or "record" not in self.meta.columns:
            return None
        cols = list(cols)
        if any(c not in self.meta.columns for c in cols):
            return None
        parts: list[np.ndarray] = []
        if cols:
            parts.append(self.meta_per_window(cols).to_numpy(np.float32))
        sex_candidates = [sex_col, *[a for a in _SEX_ALIASES if a != sex_col]]
        sex_name = next((c for c in sex_candidates if c in self.meta.columns), None)
        if sex_name is not None:
            raw = self.meta_per_window(sex_name).iloc[:, 0]
            male = raw.astype(str).str.strip().str.lower().isin(_MALE_TOKENS).to_numpy()
            parts.append(male.astype(np.float32)[:, None])
        if not parts:
            return None
        return np.hstack(parts).astype(np.float32)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        """Write the set to a compressed ``.npz`` (meta stored as JSON records)."""
        path = Path(path)
        payload: dict[str, Any] = {
            "X": self.X, "y": self.y, "subject": self.subject, "record": self.record,
            "fs": np.int64(self.fs), "source": np.array(self.source),
            "meta_json": np.array(self.meta.to_json(orient="records", date_format="iso")),
        }
        if self.time is not None:
            payload["time"] = self.time
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WindowSet":
        """Inverse of :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as d:
            meta_json = str(d["meta_json"])
            meta = pd.read_json(io.StringIO(meta_json), orient="records") if meta_json not in ("", "[]") \
                else pd.DataFrame()
            return cls(X=d["X"], y=d["y"], subject=d["subject"], record=d["record"],
                       time=d["time"] if "time" in d.files else None,
                       meta=meta, fs=int(d["fs"]), source=str(d["source"]))

    def to_dict(self) -> dict[str, Any]:
        """Lightweight JSON-able summary (counts and target statistics)."""
        return {
            "source": self.source, "fs": self.fs, "n_windows": len(self),
            "n_records": self.n_records, "n_subjects": self.n_subjects,
            "window_len": self.window_len, "has_time": self.has_time,
            "y_min": float(self.y.min()) if len(self) else None,
            "y_max": float(self.y.max()) if len(self) else None,
            "y_mean": float(self.y.mean()) if len(self) else None,
        }


def concat_windowsets(sets: Sequence[WindowSet], source: str | None = None) -> WindowSet:
    """Concatenate several sets (e.g. subjects loaded separately).

    Records must be unique across inputs.  ``time`` is kept only if every input
    has it; ``fs`` must agree.
    """
    sets = list(sets)
    if not sets:
        raise ValueError("need at least one WindowSet")
    fs = {s.fs for s in sets}
    if len(fs) != 1:
        raise ValueError(f"sampling rates differ: {sorted(fs)}")
    has_time = all(s.time is not None for s in sets)
    meta = pd.concat([s.meta for s in sets if len(s.meta)], ignore_index=True) if any(len(s.meta) for s in sets) \
        else pd.DataFrame()
    return WindowSet(
        X=np.concatenate([s.X for s in sets]), y=np.concatenate([s.y for s in sets]),
        subject=np.concatenate([s.subject for s in sets]), record=np.concatenate([s.record for s in sets]),
        time=np.concatenate([s.time for s in sets]) if has_time else None,
        meta=meta, fs=sets[0].fs, source=source if source is not None else sets[0].source,
    )


# keep dataclass field order discoverable for serialisation helpers elsewhere
FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(WindowSet))
