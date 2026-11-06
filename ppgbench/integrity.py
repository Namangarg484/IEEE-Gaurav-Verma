"""
ppgbench.integrity
==================
Data-integrity audits that catch the failure modes which invalidated the
first version of this work *before* a single model is trained.

What went wrong, and which check catches it
-------------------------------------------
* ``Archive_Dataset/clean-dataset.csv`` (a public derivative of the Mendeley
  PPG-glucose data) is a corrupted cross-join: within each subject the
  glucose label cycles row by row through that subject's reference values
  (subject 1: ``99, 102, 103, 128, 130, 134, 136, 99, 102, ...``).  Every
  multi-row window therefore carries the subject mean and the task collapses
  to predicting a per-subject constant (a per-subject oracle scores MAE 0.03 mg/dL
  with r = 1 under a random-window split of that file).
  :func:`detect_label_cycling` detects the pattern directly from the label
  column, per subject, with no knowledge of the signal.
* The same cross-join duplicates every PPG sample under several labels.
  :func:`detect_duplicate_windows` finds identical (or hop-shifted) windows,
  and :func:`audit` reports those that sit in *different* records or carry
  *conflicting* labels.
* CGM timestamps were once read in the wrong unit (ms vs s) and the guard
  gap silently became 3.6 s instead of 1 h.  :func:`check_timestamps` tests
  the spacing against the expected cadence (5 min for Dexcom G6) and flags
  unit errors, gaps, duplicates and non-monotone sequences.
* Subject / record / time leakage across folds.  :func:`assert_no_leak` and
  :func:`leak_report` are re-exported from :mod:`ppgbench.protocols`;
  :func:`audit` combines them with the data checks into one JSON-able
  report.

All functions are pure numpy/pandas and run in milliseconds on the bundled
datasets.  ``ppgbench.protocols`` is imported lazily so that the data checks
remain usable even when the rest of the package is not importable.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ppgbench.core.windowset import WindowSet

    Fold = tuple[np.ndarray, np.ndarray]

log = logging.getLogger("ppgbench.integrity")

__all__ = [
    "detect_label_cycling", "detect_duplicate_windows", "check_timestamps", "check_record_labels",
    "assert_no_leak", "leak_report", "LeakError", "audit", "audit_ok",
]


# --------------------------------------------------------------------------- #
# Lazy re-exports from ppgbench.protocols
# --------------------------------------------------------------------------- #
def assert_no_leak(ws: "WindowSet", folds: Sequence["Fold"], level: str, **kw: Any) -> None:
    """Re-export of :func:`ppgbench.protocols.assert_no_leak` (raises ``LeakError`` on a leak)."""
    from ppgbench.protocols import assert_no_leak as _impl
    return _impl(ws, folds, level, **kw)


def leak_report(ws: "WindowSet", folds: Sequence["Fold"], protocol: str | None = None, **kw: Any) -> dict[str, Any]:
    """Re-export of :func:`ppgbench.protocols.leak_report` (never raises on a leak)."""
    from ppgbench.protocols import leak_report as _impl
    return _impl(ws, folds, protocol, **kw)


def __getattr__(name: str) -> Any:
    """Lazy ``LeakError`` so ``from ppgbench.integrity import LeakError`` works without an eager import."""
    if name == "LeakError":
        from ppgbench.protocols import LeakError
        return LeakError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Label cycling (the clean-dataset.csv artefact)
# --------------------------------------------------------------------------- #
def _fraction_matching(lab: np.ndarray, k: int) -> float:
    """Fraction of rows ``i >= k`` with ``lab[i] == lab[i - k]`` (NaN never matches)."""
    if k >= len(lab):
        return 0.0
    a, b = lab[k:], lab[:-k]
    if lab.dtype.kind == "f":
        eq = np.isclose(a, b, rtol=0.0, atol=1e-9) & np.isfinite(a) & np.isfinite(b)
    else:
        eq = a == b
    return float(eq.mean())


def _cycling_1d(lab: np.ndarray, max_period: int, threshold: float, min_rows: int) -> dict[str, Any]:
    n = len(lab)
    n_unique = int(pd.unique(lab).size)
    res: dict[str, Any] = {"n": n, "n_unique": n_unique, "period": None, "fraction_matching": None,
                           "kind": None, "fractions": {}}
    if n < min_rows:
        res["kind"] = "too_short"
        return res
    for k in range(1, min(max_period, n - 1) + 1):
        f = _fraction_matching(lab, k)
        res["fractions"][k] = f
        if f > threshold:
            res["period"], res["fraction_matching"] = k, f
            res["kind"] = "constant" if k == 1 or n_unique == 1 else "cycling"
            break
    return res


def detect_label_cycling(labels: np.ndarray | Sequence[Any], max_period: int = 32, threshold: float = 0.98,
                         groups: np.ndarray | Sequence[Any] | None = None, min_rows: int = 8) -> dict[str, Any]:
    """Detect a label column that repeats with a fixed period, row by row.

    The corrupted derivative CSV paired every PPG row with every reference
    value of the same subject in turn, so ``label[i] == label[i - k]`` holds
    for (almost) every row with ``k`` = number of distinct values of that
    subject.  Genuine data never looks like this: consecutive windows of a
    record share one label (runs, period-1 matches well below 1 because the
    label changes at every record boundary), and the labels of successive
    records are unrelated.

    Parameters
    ----------
    labels : (N,) array-like
        Label per row, in file / window order.
    max_period : int
        Largest period tried (``1 .. max_period``).
    threshold : float
        A period ``k`` is reported when the fraction of rows with
        ``label[i] == label[i-k]`` exceeds it (0.98 tolerates the few rows at
        subject boundaries).
    groups : (N,) array-like, optional
        Group key (subject).  The artefact has a *different* period per
        subject, so on a multi-subject column the global test fails while
        the per-group test fires; with ``groups`` the check is run globally
        and per group.
    min_rows : int
        Groups with fewer rows are skipped (``kind="too_short"``).

    Returns
    -------
    dict with
        ``period`` (smallest period over the threshold, ``None`` if none; 1
        means the column is constant / run-structured, not cycling),
        ``fraction_matching`` (its match fraction), ``kind``
        (``"cycling"`` | ``"constant"`` | ``None``), ``n``, ``n_unique``,
        ``fractions`` (``{k: fraction}`` for the periods tried), ``threshold``,
        ``cycling`` (bool: a cycling pattern was found globally or in any
        group) and, with ``groups``, ``per_group`` (``{group: sub-dict}``),
        ``n_groups_cycling`` and ``groups_cycling`` (their keys).
    """
    lab = np.asarray(labels).reshape(-1)
    out = _cycling_1d(lab, max_period, threshold, min_rows)
    out["threshold"] = float(threshold)
    out["cycling"] = out["kind"] == "cycling"
    if groups is not None:
        g = np.asarray(groups).reshape(-1)
        if g.shape[0] != lab.shape[0]:
            raise ValueError(f"groups has {g.shape[0]} rows but labels has {lab.shape[0]}")
        per: dict[str, dict[str, Any]] = {}
        for key in pd.unique(g):
            sub = _cycling_1d(lab[g == key], max_period, threshold, min_rows)
            sub.pop("fractions")
            per[str(key)] = sub
        bad = [k for k, v in per.items() if v["kind"] == "cycling"]
        out.update(per_group=per, n_groups=len(per), n_groups_cycling=len(bad), groups_cycling=bad,
                   cycling=out["cycling"] or bool(bad))
    if out["cycling"]:
        log.warning("label cycling detected (period %s, %s of %s groups): labels look like a cross-join artefact",
                    out["period"], out.get("n_groups_cycling", "-"), out.get("n_groups", "-"))
    return out


def detect_label_cycling_local(labels: np.ndarray | Sequence[Any], groups: np.ndarray | Sequence[Any] | None = None,
                               block: int = 256, step: int | None = None, max_period: int = 32,
                               threshold: float = 0.98) -> dict[str, Any]:
    """Block-wise version of :func:`detect_label_cycling` for *partial* corruption.

    The whole-column test dilutes a corrupted stretch with clean rows: with a
    cycle of period ``k`` over a fraction ``p`` of a subject's rows, the
    period-``k`` match fraction is roughly ``p + (1 - p)(1 - k / run)`` and
    can stay under the threshold.  Scanning contiguous blocks of ``block``
    rows (stride ``step``, default ``block // 2``) restores sensitivity: any
    block lying inside the corrupted stretch cycles exactly.  On clean,
    record-structured columns a block is a single run or spans a run
    boundary, so its period-1 match fraction is at least ``1 - 1/block`` and
    it is classified as ``constant``, never as ``cycling``; ``block`` must
    therefore exceed ``1 / (1 - threshold)`` (50 rows at 0.98) and
    ``max_period``.

    Returns ``{"cycling": bool, "groups_cycling": [...], "n_groups": int,
    "n_groups_cycling": int, "first_block": {group: start_row}}``.
    """
    if block <= max(max_period, int(np.ceil(1.0 / (1.0 - threshold)))):
        raise ValueError("block must exceed max_period and 1/(1-threshold)")
    step = step or block // 2
    lab = np.asarray(labels).reshape(-1)
    g = np.zeros(len(lab), dtype=int) if groups is None else np.asarray(groups).reshape(-1)
    flagged, first = [], {}
    keys = pd.unique(g)
    for key in keys:
        col = lab[g == key]
        starts = range(0, max(1, len(col) - block + 1), step)
        for st in starts:
            res = _cycling_1d(col[st:st + block], max_period, threshold, min_rows=min(block, 8))
            if res["kind"] == "cycling":
                flagged.append(str(key)); first[str(key)] = int(st)
                break
    if flagged:
        log.warning("local label cycling detected in %d of %d groups", len(flagged), len(keys))
    return {"cycling": bool(flagged), "groups_cycling": flagged, "n_groups": int(len(keys)),
            "n_groups_cycling": len(flagged), "first_block": first, "block": block, "threshold": threshold}


# --------------------------------------------------------------------------- #
# Duplicate / overlapping windows
# --------------------------------------------------------------------------- #
def _zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(1, keepdims=True)
    sd[sd == 0] = 1.0                       # flat rows stay flat; no epsilon, so scaling is exactly invariant
    return (X - X.mean(1, keepdims=True)) / sd


def _digests(X: np.ndarray, tol: float) -> list[bytes]:
    """Content hash of each row after quantisation to ``tol`` (rows equal within ``tol`` collide)."""
    q = np.round(np.asarray(X, dtype=np.float64) / tol).astype(np.int64)
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in np.ascontiguousarray(q)]


def _pairs_from_digests(d_a: list[bytes], d_b: list[bytes] | None = None) -> list[tuple[int, int]]:
    """``(i, j)`` with equal digests; ``i < j`` within one list, or (i in a, j in b) across two lists."""
    where: dict[bytes, list[int]] = {}
    if d_b is None:
        for i, h in enumerate(d_a):
            where.setdefault(h, []).append(i)
        return [(a, b) for idx in where.values() if len(idx) > 1 for n, a in enumerate(idx) for b in idx[n + 1:]]
    for j, h in enumerate(d_b):
        where.setdefault(h, []).append(j)
    return [(i, j) for i, h in enumerate(d_a) for j in where.get(h, ()) if i != j]


def detect_duplicate_windows(X: np.ndarray, tol: float = 1e-6, hop: int | None = None,
                             normalize: bool = True) -> list[tuple[int, int]]:
    """Find windows that are exact duplicates (and, with ``hop``, shifted overlaps) of one another.

    Hash based (``O(N)``): rows are quantised to steps of ``tol`` and hashed,
    so bit-identical rows (the cross-join case) always collide and rows that
    differ by much less than ``tol`` almost always do (a sample lying within
    the perturbation of a quantisation boundary breaks the match, probability
    about ``perturbation / tol`` per sample; choose ``tol`` well above the
    noise you want to ignore).  With ``hop`` the tail
    ``X[i, hop:]`` of each window is compared with the head ``X[j, :-hop]``
    of every other, which recovers windows that were cut from the same
    signal ``hop`` samples apart (e.g. 50 % overlap).  Because windows are
    z-scored individually, both segments are re-standardised before hashing
    when ``normalize``; this makes the test scale-invariant and robust to
    the per-window normalisation.

    Returns a sorted list of index pairs ``(i, j)``; exact duplicates have
    ``i < j``, shifted pairs are ordered (earlier window, later window).
    The caller decides which pairs are suspicious: consecutive windows of one
    record overlap *by construction*; the same window in two records or
    under two labels does not.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be (N, L); got {X.shape}")
    n, L = X.shape
    if n < 2:
        return []
    Z = _zscore(X) if normalize else X.astype(np.float64)
    pairs = set(_pairs_from_digests(_digests(Z, tol)))
    if hop:
        if not 0 < hop < L:
            raise ValueError(f"hop must be in (0, {L}); got {hop}")
        tail, head = X[:, hop:], X[:, :-hop]
        if normalize:
            tail, head = _zscore(tail), _zscore(head)
        pairs.update(_pairs_from_digests(_digests(tail, tol), _digests(head, tol)))
    return sorted(pairs)


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def _timestamp_stats(t: np.ndarray, expected: float, tol: float) -> dict[str, Any]:
    t = np.asarray(t, dtype=np.float64)
    t = t[np.isfinite(t)]
    n = int(t.size)
    out: dict[str, Any] = {"n": n, "monotone": None, "n_backwards": 0, "n_duplicates": 0, "median_spacing_s": None,
                           "fraction_regular": None, "n_gaps": 0, "max_gap_s": None, "span_s": None,
                           "unit_suspect": None}
    if n < 2:
        out["monotone"] = True
        return out
    d = np.diff(t)
    out["monotone"] = bool((d >= 0).all())
    out["n_backwards"] = int((d < 0).sum())
    ts = np.sort(t)
    ds = np.diff(ts)
    out["n_duplicates"] = int((ds == 0).sum())
    nz = ds[ds > 0]
    med = float(np.median(nz)) if nz.size else 0.0
    out["median_spacing_s"] = med
    out["fraction_regular"] = float((np.abs(ds - expected) <= tol).mean())
    gaps = ds[ds > 2 * expected + tol]
    out["n_gaps"] = int(gaps.size)
    out["max_gap_s"] = float(ds.max())
    out["span_s"] = float(ts[-1] - ts[0])
    # ms, us or ns instead of s (or s instead of ms) produce spacings off by a power of 1000
    ratio = med / expected if expected > 0 and med > 0 else 1.0
    out["unit_suspect"] = None
    for factor, name in ((1e3, "milliseconds"), (1e6, "microseconds"), (1e9, "nanoseconds")):
        if 0.5 < ratio / factor < 2:
            out["unit_suspect"] = f"spacing looks like {name}, not seconds (x{factor:g})"
        if 0.5 < ratio * factor < 2:
            out["unit_suspect"] = f"spacing is {factor:g}x too small; are the times in seconds?"
    return out


def check_timestamps(t: np.ndarray | Sequence[float], expected_spacing_s: float = 300.0, tol: float = 5.0,
                     groups: np.ndarray | Sequence[Any] | None = None,
                     min_fraction_regular: float = 0.5) -> dict[str, Any]:
    """Sanity-check reference timestamps (POSIX seconds) against an expected cadence.

    For a CGM every reading should follow the previous one by
    ``expected_spacing_s`` (300 s for Dexcom G6) within ``tol`` seconds,
    apart from sensor gaps.  The check reports, overall and per ``groups``
    (subject): monotonicity, duplicates, the median spacing, the fraction of
    spacings that are regular, the number of gaps (> 2x expected), the span,
    and a ``unit_suspect`` message when the median spacing is off by a power
    of 1000 (seconds confused with milliseconds etc.).

    Duplicate times are expected when several windows share one reference
    reading, so pass the *record-level* times (``ws.meta.egv_time``) or
    de-duplicate per record for the regularity statistics; :func:`audit`
    does so.

    Returns a dict with the overall statistics, ``per_group`` (if given) and
    ``ok``: no backwards steps, no unit suspicion, and at least
    ``min_fraction_regular`` regular spacings in every group.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    out = _timestamp_stats(t, expected_spacing_s, tol)
    out.update(expected_spacing_s=float(expected_spacing_s), tol_s=float(tol))
    problems: list[str] = []
    stats = [out]
    if groups is not None:
        g = np.asarray(groups).reshape(-1)
        if g.shape[0] != t.shape[0]:
            raise ValueError(f"groups has {g.shape[0]} rows but t has {t.shape[0]}")
        out["per_group"] = {str(k): _timestamp_stats(t[g == k], expected_spacing_s, tol) for k in pd.unique(g)}
        stats = list(out["per_group"].values())          # regularity / order are per-subject properties
    for i, s in enumerate(stats):
        who = "" if groups is None else f" (group {list(out['per_group'])[i]})"
        if s["n"] < 2:
            continue
        if s["n_backwards"]:
            problems.append(f"{s['n_backwards']} backwards step(s){who}")
        if s["unit_suspect"]:
            problems.append(f"{s['unit_suspect']}{who}")
        if s["fraction_regular"] is not None and s["fraction_regular"] < min_fraction_regular:
            problems.append(f"only {s['fraction_regular']:.0%} of spacings are {expected_spacing_s:g}+-{tol:g} s{who}")
    out["problems"] = problems
    out["ok"] = not problems
    if problems:
        log.warning("timestamp check: %s", "; ".join(problems[:5]))
    return out


# --------------------------------------------------------------------------- #
# Record / label consistency
# --------------------------------------------------------------------------- #
def check_record_labels(ws: "WindowSet") -> dict[str, Any]:
    """Every record must carry exactly one label and belong to exactly one subject, and agree with ``meta``.

    A record is one reference measurement; two labels under one record id
    (or one record under two subjects) means the join between signals and
    labels is broken.  ``meta.glucose`` is compared to the window labels when
    present.  Returns ``ok`` plus the offending record ids (first 10).
    """
    df = pd.DataFrame({"record": np.asarray(ws.record), "subject": np.asarray(ws.subject),
                       "y": np.asarray(ws.y, dtype=np.float64)})
    nl = df.groupby("record").y.nunique()
    ns = df.groupby("record").subject.nunique()
    multi_label = nl.index[nl > 1].tolist()
    multi_subject = ns.index[ns > 1].tolist()
    meta_mismatch: list[str] = []
    meta = getattr(ws, "meta", None)
    if meta is not None and len(meta) and {"record", "glucose"} <= set(meta.columns):
        m = meta.assign(record=meta["record"].astype(str)).set_index("record")["glucose"].astype(float)
        first = df.groupby("record").y.first()
        common = first.index.intersection(m.index)
        bad = ~np.isclose(first.loc[common].to_numpy(), m.loc[common].to_numpy(), atol=1e-4)
        meta_mismatch = [str(r) for r in common[bad]]
    return {"n_records": int(nl.size), "n_multi_label_records": len(multi_label),
            "multi_label_records": [str(r) for r in multi_label[:10]],
            "n_multi_subject_records": len(multi_subject),
            "multi_subject_records": [str(r) for r in multi_subject[:10]],
            "n_meta_mismatch": len(meta_mismatch), "meta_mismatch": meta_mismatch[:10],
            "ok": not (multi_label or multi_subject or meta_mismatch)}


# --------------------------------------------------------------------------- #
# Combined audit
# --------------------------------------------------------------------------- #
def _record_times(ws: "WindowSet") -> tuple[np.ndarray, np.ndarray] | None:
    """(time, subject) with one entry per record, or ``None`` when the set has no times."""
    t = getattr(ws, "time", None)
    meta = getattr(ws, "meta", None)
    if t is None and meta is not None and len(meta) and "record" in meta.columns:
        for col in ("egv_time", "time"):
            if col in meta.columns:
                m = meta.assign(record=meta["record"].astype(str)).set_index("record")
                t = m[col].loc[np.asarray(ws.record)].to_numpy(np.float64)
                break
    if t is None:
        return None
    df = pd.DataFrame({"record": np.asarray(ws.record), "subject": np.asarray(ws.subject),
                       "t": np.asarray(t, dtype=np.float64)})
    rec = df.groupby("record", sort=False).agg(subject=("subject", "first"), t=("t", "first"))
    rec = rec.sort_values(["subject", "t"], kind="stable")
    return rec.t.to_numpy(), rec.subject.to_numpy()


def audit(ws: "WindowSet", folds: Sequence["Fold"] | None = None, protocol: str | None = None, *,
          hop: int | None = None, expected_spacing_s: float = 300.0, time_tol_s: float = 5.0,
          max_period: int = 32, level: str | None = None, gap_s: float | None = None) -> dict[str, Any]:
    """Run every integrity check on a :class:`WindowSet` (and its folds) and return one JSON-able report.

    Sections
    --------
    ``labels``      :func:`detect_label_cycling` on the window labels grouped
                    by subject and on the record-level ``meta.glucose``.
    ``records``     :func:`check_record_labels`.
    ``duplicates``  :func:`detect_duplicate_windows` (with ``hop``: shifted
                    overlaps too) split into ``cross_record`` pairs (the
                    same signal under two record ids) and
                    ``conflicting_label`` pairs (identical signal, different
                    label), which are the suspicious ones; ``within_record``
                    overlaps are expected with overlapping windowing.
    ``timestamps``  :func:`check_timestamps` on record-level times per
                    subject (``None`` when the set has no times).
    ``leak``        :func:`leak_report` for ``folds`` plus ``assert_no_leak``
                    at ``level`` (default: the protocol's declared level)
                    with the guard gap ``gap_s`` (default: the protocol's;
                    pass the gap the folds were built with), with the error
                    message captured instead of raised -- including the
                    ``ValueError`` for NaN/inf reference times, which can
                    never pass the ``time`` level.
    ``ok``          ``True`` when no section reports a problem.
    """
    rep: dict[str, Any] = {"source": getattr(ws, "source", ""), "n_windows": int(len(ws)),
                           "n_records": int(len(np.unique(ws.record))),
                           "n_subjects": int(len(np.unique(ws.subject))), "protocol": protocol}
    problems: list[str] = []

    lab = detect_label_cycling(ws.y, max_period=max_period, groups=ws.subject)
    lab.pop("fractions", None)
    rep["labels"] = {"windows": lab}
    if lab["cycling"]:
        problems.append(f"label cycling in {lab['n_groups_cycling']} subject(s) (window labels)")
    meta = getattr(ws, "meta", None)
    if meta is not None and len(meta) and {"subject", "glucose"} <= set(meta.columns):
        lab_r = detect_label_cycling(meta["glucose"].to_numpy(), max_period=max_period,
                                     groups=meta["subject"].to_numpy())
        lab_r.pop("fractions", None)
        rep["labels"]["records"] = lab_r
        if lab_r["cycling"]:
            problems.append(f"label cycling in {lab_r['n_groups_cycling']} subject(s) (record labels)")

    rec = check_record_labels(ws)
    rep["records"] = rec
    if not rec["ok"]:
        problems.append("records with several labels/subjects or meta mismatch")

    pairs = detect_duplicate_windows(ws.X, hop=hop)
    record, y = np.asarray(ws.record), np.asarray(ws.y)
    cross = [(i, j) for i, j in pairs if record[i] != record[j]]
    conflict = [(i, j) for i, j in pairs if y[i] != y[j]]
    rep["duplicates"] = {"n_pairs": len(pairs), "n_within_record": len(pairs) - len(cross),
                         "n_cross_record": len(cross), "n_conflicting_label": len(conflict),
                         "cross_record": [(int(i), int(j)) for i, j in cross[:20]],
                         "conflicting_label": [(int(i), int(j)) for i, j in conflict[:20]], "hop": hop}
    if cross:
        problems.append(f"{len(cross)} duplicate window pair(s) across records")

    rt = _record_times(ws)
    if rt is None:
        rep["timestamps"] = None
    else:
        ts = check_timestamps(rt[0], expected_spacing_s, time_tol_s, groups=rt[1])
        rep["timestamps"] = ts
        if not ts["ok"]:
            problems.append("timestamps: " + "; ".join(ts["problems"][:3]))

    if folds is not None:
        try:
            from ppgbench import protocols as P
            kw = {} if gap_s is None else {"gap_s": gap_s}
            lr = P.leak_report(ws, folds, protocol, **kw)
            lvl = level or lr.get("declared_level") or "subject"
            try:
                P.assert_no_leak(ws, folds, lvl, gap_s=float(lr.get("gap_s", 0.0)), protocol=protocol)
                leak_err = None
            except (P.LeakError, ValueError) as e:      # ValueError: level 'time' with NaN/inf reference times
                leak_err = str(e)
            rep["leak"] = {"level": lvl, "gap_s": lr.get("gap_s"), "passes": lr["passes"], "totals": lr["totals"],
                           "n_folds": lr["n_folds"], "time_finite": lr.get("time_finite"),
                           "test_coverage": lr["test_coverage"], "test_sets_disjoint": lr["test_sets_disjoint"],
                           "fingerprint": lr.get("fingerprint"), "error": leak_err}
            if leak_err:
                problems.append(f"leak at level {lvl}: {leak_err.splitlines()[0]}")
        except ImportError as e:  # protocols module unavailable (partial checkout)
            rep["leak"] = {"error": f"ppgbench.protocols unavailable: {e}"}
            problems.append("leak check unavailable")
    else:
        rep["leak"] = None

    rep["problems"] = problems
    rep["ok"] = not problems
    (log.warning if problems else log.info)("integrity audit of %s: %s", rep["source"] or "WindowSet",
                                            "; ".join(problems) if problems else "clean")
    return rep


def audit_ok(ws: "WindowSet", folds: Sequence["Fold"] | None = None, protocol: str | None = None, **kw: Any) -> bool:
    """``audit(...)["ok"]`` -- convenience for ``assert audit_ok(ws, folds, protocol)``."""
    return bool(audit(ws, folds, protocol, **kw)["ok"])
