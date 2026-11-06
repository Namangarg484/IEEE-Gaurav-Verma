"""
ppgbench.data.registry
======================
Named, discoverable datasets with a single entry point.

Why a registry
--------------
A benchmark is only credible when every number can be traced back to *which*
data were used, with *which* loader options.  Hard-coding a loader call in each
experiment script makes that provenance implicit and easy to break.  Here every
dataset is a named loader that returns a :class:`~ppgbench.core.windowset.WindowSet`
and is reachable through :func:`load_dataset`::

    from ppgbench.data import load_dataset, list_datasets
    ws = load_dataset("mendeley")                       # raw Mendeley recordings
    ws = load_dataset("bigideas", subjects=["001"])     # PhysioNet BIG IDEAs
    ws = load_dataset("synthetic", seed=3)              # deterministic test data

Every loaded set is checked by :func:`validate_windowset` (shape, dtype,
record/subject consistency, time alignment), so a corrupted file or a loader
bug surfaces at load time rather than as an implausibly good metric three
scripts later.  :func:`fingerprint` returns a short content hash so result
stores can record exactly which array a metric was computed on.

Adding a dataset
----------------
Decorate a loader that returns a ``WindowSet``::

    @register_dataset("mydata", description="...", citation="...")
    def load_mydata(root: str | None = None, win: int = 320) -> WindowSet: ...

Loader keyword arguments are introspected and validated, so
``load_dataset("mydata", wim=320)`` raises a ``TypeError`` naming the valid
options instead of being silently ignored.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ppgbench.core.windowset import WindowSet

log = logging.getLogger("ppgbench.data.registry")

__all__ = [
    "FS", "WIN", "HOP",
    "DatasetError", "DatasetInfo",
    "register_dataset", "load_dataset", "list_datasets", "dataset_info",
    "is_available", "describe_datasets", "validate_windowset", "fingerprint",
    "clear_cache",
]

# --------------------------------------------------------------------------- #
# Common representation shared by all loaders (identical to ppg_data.py)
# --------------------------------------------------------------------------- #
FS: int = 64            #: common sampling rate (Hz) == Empatica E4 BVP
WIN: int = 5 * FS       #: 320 samples = 5-s analysis window
HOP: int = WIN // 2     #: 50 % overlap for short recordings

LoaderFn = Callable[..., WindowSet]


class DatasetError(ValueError):
    """A dataset failed validation or could not be located."""


@dataclass(frozen=True)
class DatasetInfo:
    """Static description of a registered dataset.

    Attributes
    ----------
    name : registry key (``load_dataset(name)``).
    loader : callable returning a ``WindowSet``.
    description : one-paragraph description of the source and its labels.
    citation : how to cite the dataset in a paper.
    license : data licence (informational).
    url : where the raw data can be obtained.
    requires_files : ``False`` for datasets generated in memory.
    availability : optional zero-argument callable reporting whether the raw
        files are present on this machine.
    tags : free-form labels (``"cgm"``, ``"spot"``, ``"synthetic"`` ...).
    """

    name: str
    loader: LoaderFn
    description: str = ""
    citation: str = ""
    license: str = ""
    url: str = ""
    requires_files: bool = True
    availability: Callable[[], bool] | None = field(default=None, compare=False)
    tags: tuple[str, ...] = ()

    @property
    def params(self) -> dict[str, Any]:
        """Loader keyword arguments and their defaults (``inspect.Parameter.empty`` if required)."""
        sig = inspect.signature(self.loader)
        return {k: p.default for k, p in sig.parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

    @property
    def accepts_var_kwargs(self) -> bool:
        return any(p.kind == p.VAR_KEYWORD for p in inspect.signature(self.loader).parameters.values())

    def is_available(self) -> bool:
        """``True`` when the loader can be expected to succeed on this machine."""
        if not self.requires_files:
            return True
        if self.availability is None:
            return True            # unknown: assume present, loader will raise otherwise
        try:
            return bool(self.availability())
        except Exception as exc:   # pragma: no cover - defensive
            log.debug("availability check for %s failed: %s", self.name, exc)
            return False

    def summary(self) -> str:
        """One-line human description used by ``python -m ppgbench datasets``."""
        avail = "available" if self.is_available() else "missing"
        params = ", ".join(f"{k}={v!r}" if v is not inspect.Parameter.empty else k
                           for k, v in self.params.items())
        return f"{self.name:<10} [{avail}] {self.description.strip().splitlines()[0] if self.description else ''}" \
               f"\n{'':<10}  kwargs: {params}"


_REGISTRY: dict[str, DatasetInfo] = {}
_CACHE: dict[tuple[str, str], WindowSet] = {}


# --------------------------------------------------------------------------- #
# Registration and lookup
# --------------------------------------------------------------------------- #
def register_dataset(name: str, *, description: str = "", citation: str = "", license: str = "",
                     url: str = "", requires_files: bool = True,
                     availability: Callable[[], bool] | None = None,
                     tags: tuple[str, ...] | list[str] = (), overwrite: bool = False) -> Callable[[LoaderFn], LoaderFn]:
    """Decorator registering a loader under ``name``.

    Parameters mirror :class:`DatasetInfo`.  Registering an existing name
    raises ``ValueError`` unless ``overwrite=True`` (useful in tests and
    notebooks that monkey-patch a loader).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("dataset name must be a non-empty string")

    def decorator(fn: LoaderFn) -> LoaderFn:
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"dataset {name!r} is already registered (use overwrite=True to replace it)")
        _REGISTRY[name] = DatasetInfo(name=name, loader=fn, description=description or (fn.__doc__ or ""),
                                      citation=citation, license=license, url=url,
                                      requires_files=requires_files, availability=availability,
                                      tags=tuple(tags))
        setattr(fn, "dataset_name", name)
        log.debug("registered dataset %s -> %s.%s", name, fn.__module__, fn.__qualname__)
        return fn

    return decorator


def list_datasets() -> list[str]:
    """Sorted names of all registered datasets."""
    return sorted(_REGISTRY)


def dataset_info(name: str) -> DatasetInfo:
    """Return the :class:`DatasetInfo` for ``name`` or raise ``KeyError`` listing the alternatives."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; registered: {list_datasets()}") from None


def is_available(name: str) -> bool:
    """Whether the raw files for ``name`` are present (always ``True`` for in-memory datasets)."""
    return dataset_info(name).is_available()


def describe_datasets() -> str:
    """Multi-line table of every registered dataset, its availability and loader options."""
    return "\n".join(info.summary() for info in (_REGISTRY[n] for n in list_datasets()))


def _check_kwargs(info: DatasetInfo, kwargs: dict[str, Any]) -> None:
    if info.accepts_var_kwargs:
        return
    unknown = sorted(set(kwargs) - set(info.params))
    if unknown:
        raise TypeError(f"dataset {info.name!r} does not accept {unknown}; valid kwargs: {sorted(info.params)}")


def _cache_key(name: str, kwargs: dict[str, Any]) -> tuple[str, str]:
    return name, repr(sorted((k, repr(v)) for k, v in kwargs.items()))


def load_dataset(name: str, *, validate: bool = True, cache: bool = False, **kwargs: Any) -> WindowSet:
    """Load a registered dataset as a validated ``WindowSet``.

    Parameters
    ----------
    name : registry key, see :func:`list_datasets`.
    validate : run :func:`validate_windowset` on the result (recommended; cheap).
    cache : keep the loaded set in an in-process cache keyed on ``(name, kwargs)``
        so an ``audit`` followed by a ``run`` on the same data loads once.  The
        cached object is returned as-is; do not mutate it.
    **kwargs : forwarded to the loader after being checked against its signature.
    """
    info = dataset_info(name)
    _check_kwargs(info, kwargs)
    key = _cache_key(name, kwargs)
    if cache and key in _CACHE:
        log.info("dataset %s served from cache", name)
        return _CACHE[key]
    log.info("loading dataset %s %s", name, kwargs or "")
    ws = info.loader(**kwargs)
    if not isinstance(ws, WindowSet):
        raise DatasetError(f"loader for {name!r} returned {type(ws).__name__}, expected WindowSet")
    if validate:
        validate_windowset(ws)
    log.info("%s  fingerprint %s", ws.describe(), fingerprint(ws))
    if cache:
        _CACHE[key] = ws
    return ws


def clear_cache() -> None:
    """Drop every cached ``WindowSet``."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Validation and provenance
# --------------------------------------------------------------------------- #
def validate_windowset(ws: WindowSet, strict: bool = True) -> list[str]:
    """Check the structural invariants every protocol and metric relies on.

    The checks are the ones whose violation silently corrupts a benchmark:
    mismatched array lengths (labels shifted against windows), a record that
    maps to two subjects (subject-wise splits leak), a record with two
    different glucose values or two timestamps (record-level aggregation
    averages unrelated references), and non-finite windows (NaNs propagate
    into every metric).  Per-window z-scoring is reported as a warning only,
    because a legitimately flat window z-scores to all-zeros.

    Returns the list of problems; raises :class:`DatasetError` when
    ``strict`` and the list is non-empty.
    """
    errs: list[str] = []
    X, y, subject, record, t = ws.X, ws.y, ws.subject, ws.record, ws.time
    n = len(y)
    if X.ndim != 2:
        errs.append(f"X must be 2-D (N, L), got shape {X.shape}")
    if X.dtype != np.float32:
        errs.append(f"X must be float32, got {X.dtype}")
    if y.ndim != 1 or y.dtype != np.float32:
        errs.append(f"y must be 1-D float32, got shape {y.shape} dtype {y.dtype}")
    for nm, arr in (("subject", subject), ("record", record)):
        if arr.ndim != 1:
            errs.append(f"{nm} must be 1-D, got shape {arr.shape}")
        if arr.dtype.kind not in "UOS":
            errs.append(f"{nm} must hold strings, got dtype {arr.dtype}")
    if n == 0:
        errs.append("empty WindowSet (no windows)")
    for nm, arr in (("X", X), ("subject", subject), ("record", record)):
        if len(arr) != n:
            errs.append(f"len({nm})={len(arr)} != len(y)={n}")
    if t is not None:
        if t.ndim != 1 or len(t) != n:
            errs.append(f"time must be 1-D of length {n}, got shape {t.shape}")
        elif t.dtype != np.float64:
            errs.append(f"time must be float64 POSIX seconds, got {t.dtype}")
        elif not np.isfinite(t).all():
            errs.append("time contains non-finite values")
    if X.size and not np.isfinite(X).all():
        errs.append(f"X contains {int((~np.isfinite(X)).sum())} non-finite values")
    if y.size and not np.isfinite(y).all():
        errs.append("y contains non-finite values")
    if not isinstance(ws.fs, (int, np.integer)) or ws.fs <= 0:
        errs.append(f"fs must be a positive int, got {ws.fs!r}")
    if errs:                                    # structural errors: stop before relational checks
        return _finish(errs, strict)

    # relational checks: record -> subject, record -> glucose, record -> time
    df = pd.DataFrame({"record": record, "subject": subject, "y": y})
    multi_subj = df.groupby("record").subject.nunique()
    bad = multi_subj[multi_subj > 1]
    if len(bad):
        errs.append(f"{len(bad)} record(s) map to more than one subject, e.g. {bad.index[:3].tolist()}")
    multi_y = df.groupby("record").y.nunique()
    bad = multi_y[multi_y > 1]
    if len(bad):
        errs.append(f"{len(bad)} record(s) carry more than one target value, e.g. {bad.index[:3].tolist()}")
    if t is not None:
        multi_t = pd.Series(t).groupby(record).nunique()
        bad = multi_t[multi_t > 1]
        if len(bad):
            errs.append(f"{len(bad)} record(s) carry more than one timestamp, e.g. {bad.index[:3].tolist()}")

    meta = ws.meta
    if not isinstance(meta, pd.DataFrame):
        errs.append(f"meta must be a DataFrame, got {type(meta).__name__}")
        return _finish(errs, strict)
    missing_cols = [c for c in ("record", "subject", "glucose") if c not in meta.columns]
    if missing_cols:
        errs.append(f"meta lacks required columns {missing_cols}")
        return _finish(errs, strict)
    if meta.record.duplicated().any():
        errs.append(f"meta has duplicated record rows: {meta.record[meta.record.duplicated()].head(3).tolist()}")
    recs = set(np.unique(record).tolist())
    meta_recs = set(meta.record.astype(str).tolist())
    if recs != meta_recs:
        only_arr, only_meta = sorted(recs - meta_recs)[:3], sorted(meta_recs - recs)[:3]
        errs.append(f"meta records != window records (missing from meta: {only_arr}, extra in meta: {only_meta})")
    else:
        m = meta.set_index(meta.record.astype(str))
        first = df.groupby("record").first()
        subj_meta = m.loc[first.index, "subject"].astype(str).to_numpy()
        if not np.array_equal(subj_meta, first.subject.astype(str).to_numpy()):
            errs.append("meta.subject disagrees with the window subject array for some records")
        g_meta = m.loc[first.index, "glucose"].to_numpy(np.float64)
        if not np.allclose(g_meta, first.y.to_numpy(np.float64), rtol=1e-5, atol=1e-3):
            errs.append("meta.glucose disagrees with y for some records")

    # soft checks
    if X.shape[1] > 1:
        mu = np.abs(X.mean(1)).max()
        sd = X.std(1)
        if mu > 1e-3 or np.any((sd > 1e-6) & (np.abs(sd - 1) > 1e-2)):
            log.warning("validate_windowset: windows do not look z-scored per row (|mean| max %.3g)", mu)
    return _finish(errs, strict)


def _finish(errs: list[str], strict: bool) -> list[str]:
    if errs and strict:
        raise DatasetError("WindowSet failed validation:\n  - " + "\n  - ".join(errs))
    for e in errs:
        log.warning("validate_windowset: %s", e)
    return errs


def fingerprint(ws: WindowSet, n_hex: int = 16) -> str:
    """Short SHA-256 of the arrays that determine every downstream number.

    Covers ``X``, ``y``, ``subject``, ``record`` and ``time`` (not ``meta``,
    whose informational columns may legitimately differ between loader
    versions).  Two sets with the same fingerprint yield identical folds,
    predictions and metrics for a deterministic pipeline.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ws.X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(ws.y, dtype=np.float32).tobytes())
    h.update("\x00".join(map(str, ws.subject.tolist())).encode())
    h.update(b"\x01")
    h.update("\x00".join(map(str, ws.record.tolist())).encode())
    h.update(b"\x02")
    if ws.time is not None:
        h.update(np.ascontiguousarray(ws.time, dtype=np.float64).tobytes())
    h.update(str(ws.fs).encode())
    return h.hexdigest()[:n_hex]
