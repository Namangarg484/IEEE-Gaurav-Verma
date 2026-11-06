"""
ppgbench.metrics
================
Record-level regression metrics, clinical error grids and cluster-bootstrap
confidence intervals for PPG -> glucose benchmarks.

Design principles
-----------------
* **Record level, never window level.**  Overlapping windows of one record
  share a label, so they are not independent samples.  :func:`record_level`
  averages window predictions per record first; every metric is computed on
  that table (one row per independent reference measurement).  This is what
  ``run_subject_cv.recording_level`` did and what the paper reports.
* **Numerical parity with the legacy scripts.**  :func:`summarize` reproduces
  ``run_subject_cv.metrics`` operation by operation (same dtypes, same
  formulae, same Clarke classifier as ``plot_clarke_error_grid``), and
  :func:`cluster_bootstrap` reproduces ``bootstrap_ci.cluster_bootstrap``
  draw by draw.  The parity tests in ``tests/ppgbench/test_metrics.py``
  guard this.
* **Clinical context.**  A glucose regressor is judged by whether its errors
  would change a treatment decision, not only by MAE.  We report the Clarke
  error grid (Clarke et al. 1987, the historical standard) *and* the Parkes
  consensus error grid (Parkes et al. 2000), which replaced Clarke's
  discontinuous zones with boundaries agreed by 100 clinicians and is the
  grid required by ISO 15197:2013 for meter evaluation.  Hypo-/hyperglycaemia
  event detection metrics are reported as well, because zone A alone is a
  property of the cohort's glucose spread rather than of the model.
* **Honest uncertainty.**  Subjects, not records, are the exchangeable units
  (records within a subject are correlated), so every confidence interval
  resamples subjects (cluster bootstrap) and paired model-vs-null differences
  are computed on the *same* resampled subjects.
* **Degenerate inputs never raise.**  Fewer than two records or a
  zero-variance reference (e.g. a quick smoke run with one subject) return a
  dict with ``error="degenerate"`` so batch runs keep going and the report
  layer can render "--".
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only; avoid importing the core package at runtime
    from ppgbench.core.windowset import WindowSet

log = logging.getLogger("ppgbench.metrics")

__all__ = [
    "ZONES", "ZONE_INDEX", "PARKES_VERTICES",
    "mae", "rmse", "mard_pct", "r2", "pearson_r", "bias",
    "within_subject_r", "within_subject_r_median",
    "clarke_zone", "clarke_summary", "parkes_zone", "parkes_summary", "zone_counts",
    "event_metrics",
    "record_level", "summarize", "per_subject", "aggregate_seeds", "pool_seeds",
    "cluster_bootstrap", "cluster_bootstrap_many", "cluster_bootstrap_metrics", "paired_cluster_bootstrap",
    "DEGENERATE",
]

#: Error-grid zones from clinically accurate (A) to dangerous (E).
ZONES: tuple[str, ...] = ("A", "B", "C", "D", "E")
ZONE_INDEX: dict[str, int] = {z: i for i, z in enumerate(ZONES)}
#: Value of the ``error`` key returned for degenerate inputs.
DEGENERATE = "degenerate"

ArrayLike = Sequence[float] | np.ndarray | pd.Series


# --------------------------------------------------------------------------- #
# Elementary metrics
# --------------------------------------------------------------------------- #
def _pair(y: ArrayLike, p: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: ref {y.shape} vs pred {p.shape}")
    return y.reshape(-1), p.reshape(-1)


def mae(y: ArrayLike, p: ArrayLike) -> float:
    """Mean absolute error (mg/dL)."""
    y, p = _pair(y, p)
    return float(np.abs(p - y).mean())


def rmse(y: ArrayLike, p: ArrayLike) -> float:
    """Root-mean-square error (mg/dL)."""
    y, p = _pair(y, p)
    return float(math.sqrt(((p - y) ** 2).mean()))


def bias(y: ArrayLike, p: ArrayLike) -> float:
    """Mean signed error ``pred - ref`` (mg/dL); negative = under-estimation."""
    y, p = _pair(y, p)
    return float((p - y).mean())


def mard_pct(y: ArrayLike, p: ArrayLike) -> float:
    """Mean absolute relative difference in percent, ``mean(|p - y| / y) * 100``.

    MARD is the CGM industry's headline accuracy figure.  It is undefined for
    a zero reference; such rows yield ``inf``/``nan`` exactly as in the legacy
    script (references below 40 mg/dL never occur in the bundled datasets).
    """
    y, p = _pair(y, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((np.abs(p - y) / y).mean() * 100)


def r2(y: ArrayLike, p: ArrayLike) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Can be negative: a model worse than predicting the mean of the *test*
    references.  ``nan`` for zero-variance references.
    """
    y, p = _pair(y, p)
    e = p - y
    ss_res, ss_tot = float((e ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson_r(y: ArrayLike, p: ArrayLike) -> float:
    """Pearson correlation; ``nan`` when either side has zero variance (e.g. a mean predictor)."""
    y, p = _pair(y, p)
    if len(y) < 2:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.corrcoef(y, p)[0, 1])


def within_subject_r(df: pd.DataFrame, min_records: int = 3) -> pd.Series:
    """Per-subject Pearson correlation between ``y`` and ``pred``.

    A subject contributes a value only with more than two records and
    non-zero spread in both reference and prediction (otherwise ``nan``),
    exactly as ``run_subject_cv.metrics`` did.  Within-subject correlation is
    the quantity that matters for a personal sensor: a model can achieve a
    high *pooled* r purely by ranking subjects (identity regression) while
    tracking nothing inside any one of them.
    """
    out: dict[str, float] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for s, g in df.groupby("subject", sort=True):
            ok = len(g) >= min_records and g.y.std() > 0 and g.pred.std() > 0
            out[str(s)] = float(np.corrcoef(g.y, g.pred)[0, 1]) if ok else float("nan")
    return pd.Series(out, dtype=float, name="within_subject_r")


def within_subject_r_median(df: pd.DataFrame) -> float | None:
    """Median of :func:`within_subject_r` over subjects, ``None`` if no subject qualifies."""
    w = within_subject_r(df).to_numpy()
    return float(np.nanmedian(w)) if np.isfinite(w).any() else None


# --------------------------------------------------------------------------- #
# Clarke error grid (Clarke et al., Diabetes Care 1987; 10:622-628)
# --------------------------------------------------------------------------- #
def clarke_zone(ref: ArrayLike, pred: ArrayLike) -> np.ndarray:
    """Vectorised Clarke error-grid zone (``'A'``..``'E'``) for each pair.

    The decision rules are those of ``plot_clarke_error_grid._classify_point``
    applied in the same priority order (A, then E before D to resolve the
    boundary overlap, then D, C, B), so :func:`clarke_summary` matches
    ``summarize_clarke_zones`` exactly.  Zone definitions:

    A  within 20 % of the reference (or within 20 mg/dL when ref < 70);
    B  benign error, no inappropriate treatment;
    C  over-correction of an acceptable glucose (euglycaemic range);
    D  failure to detect hypo-/hyperglycaemia (dangerous silent error);
    E  erroneous treatment, opposite to what is needed.
    """
    r, p = _pair(ref, pred)
    r, p = r.astype(np.float64), p.astype(np.float64)
    d = np.abs(p - r)
    a = np.where(r < 70, d <= 20, d <= 0.20 * r)
    e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    dz = (((r >= 240) & (p <= 180)) | ((r <= 70) & (p >= 70) & (p < 180))
          | ((r >= 70) & (r <= 180) & ((p <= 70) | (p >= 240))))
    c = (r > 70) & (r < 180) & ((p > r + 110) | (p < r - 110))
    return np.select([a, e, dz, c], ["A", "E", "D", "C"], default="B").astype("<U1")


def zone_counts(zones: np.ndarray, prefix: str = "zone") -> dict[str, Any]:
    """Counts and percentages per zone, ``{prefix}_A_count``, ``{prefix}_A_pct`` ... plus ``total_points``."""
    zones = np.asarray(zones)
    n = int(zones.size)
    out: dict[str, Any] = {"total_points": n}
    for z in ZONES:
        k = int((zones == z).sum())
        out[f"{prefix}_{z}_count"] = k
        out[f"{prefix}_{z}_pct"] = float(k / n * 100) if n else 0.0
    return out


def clarke_summary(ref: ArrayLike, pred: ArrayLike) -> dict[str, Any]:
    """Clarke zone counts/percentages; identical to ``plot_clarke_error_grid.summarize_clarke_zones``."""
    return zone_counts(clarke_zone(ref, pred), "zone")


# --------------------------------------------------------------------------- #
# Parkes consensus error grid (Parkes et al., Diabetes Care 2000; 23:1143-1148)
# --------------------------------------------------------------------------- #
#: Boundary polylines of the Parkes (consensus) error grid as ``(reference, measured)``
#: vertices in mg/dL, transcribed from Table 1 of Pfützner A, Klonoff DC, Pardo S,
#: Parkes JL, "Technical aspects of the Parkes error grid", J Diabetes Sci Technol
#: 2013;7(5):1275-1281, doi:10.1177/193229681300700517, which published the exact
#: coordinates of the grid originally drawn in Parkes JL, Slatin SL, Pardo S,
#: Ginsberg BH, "A new consensus error grid to evaluate the clinical accuracy of
#: patient-performed SMBG", Diabetes Care 2000;23(8):1143-1148.
#:
#: ``upper[Z]`` is the line separating zone Z from the next-worse zone *above* the
#: diagonal (measured too high), ``lower[Z]`` the corresponding line below it.
#: Zone E exists only above the diagonal (a gross over-estimate of a hypoglycaemic
#: reference); below the diagonal zone D extends to the axis.  Lines are extended
#: linearly beyond their last vertex, as in the published grid (defined to 550).
PARKES_VERTICES: dict[int, dict[str, dict[str, tuple[tuple[float, float], ...]]]] = {
    1: {
        "upper": {
            "A": ((0, 50), (30, 50), (140, 170), (280, 380), (430, 550)),
            "B": ((0, 60), (30, 60), (50, 80), (70, 110), (260, 550)),
            "C": ((0, 100), (25, 100), (50, 125), (80, 215), (125, 550)),
            "D": ((0, 150), (35, 155), (50, 550)),
        },
        "lower": {
            "A": ((50, 0), (50, 30), (170, 145), (385, 300), (550, 450)),
            "B": ((120, 0), (120, 30), (260, 130), (550, 250)),
            "C": ((250, 0), (250, 40), (550, 150)),
        },
    },
    2: {
        "upper": {
            "A": ((0, 50), (30, 50), (230, 330), (440, 550)),
            "B": ((0, 60), (30, 60), (280, 550)),
            "C": ((0, 80), (25, 80), (35, 90), (125, 550)),
            "D": ((0, 200), (35, 200), (50, 550)),
        },
        "lower": {
            "A": ((50, 0), (50, 30), (90, 80), (330, 230), (550, 450)),
            "B": ((90, 0), (260, 130), (550, 250)),
            "C": ((250, 0), (250, 40), (410, 110), (550, 160)),
        },
    },
}


def _polyline(xv: np.ndarray, yv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the polyline through ``(xv, yv)`` at ``x`` with linear extrapolation beyond the ends."""
    y = np.interp(x, xv, yv)
    hi = x > xv[-1]
    if hi.any():
        slope = (yv[-1] - yv[-2]) / (xv[-1] - xv[-2])
        y[hi] = yv[-1] + slope * (x[hi] - xv[-1])
    lo = x < xv[0]
    if lo.any():
        slope = (yv[1] - yv[0]) / (xv[1] - xv[0])
        y[lo] = yv[0] + slope * (x[lo] - xv[0])
    return y


def parkes_zone(ref: ArrayLike, pred: ArrayLike, diabetes_type: int = 1) -> np.ndarray:
    """Vectorised Parkes consensus error-grid zone (``'A'``..``'E'``) for each pair.

    Upper boundaries are evaluated as ``measured = f(reference)`` and lower
    boundaries, which contain vertical segments, as ``reference = g(measured)``
    (the grid is a reflection of itself about the diagonal in structure, not
    in values).  A point exactly on a boundary belongs to the inner (better)
    zone.  ``diabetes_type`` selects the type-1 (default; stricter) or
    type-2 grid.  See :data:`PARKES_VERTICES` for the source of the vertices.
    """
    if diabetes_type not in PARKES_VERTICES:
        raise ValueError(f"diabetes_type must be one of {sorted(PARKES_VERTICES)}, got {diabetes_type!r}")
    r, p = _pair(ref, pred)
    r, p = r.astype(np.float64), p.astype(np.float64)
    lines = PARKES_VERTICES[diabetes_type]
    zone = np.zeros(r.shape, dtype=np.int64)                 # 0 = A
    for i, z in enumerate(("A", "B", "C", "D")):
        v = np.asarray(lines["upper"][z], dtype=np.float64)
        above = p > _polyline(v[:, 0], v[:, 1], r)
        zone = np.where(above, i + 1, zone)                   # B, C, D, E
    for i, z in enumerate(("A", "B", "C")):
        v = np.asarray(lines["lower"][z], dtype=np.float64)
        right = r > _polyline(v[:, 1], v[:, 0], p)            # x = g(y)
        zone = np.where(right, i + 1, zone)                   # B, C, D
    return np.asarray(ZONES, dtype="<U1")[zone]


def parkes_summary(ref: ArrayLike, pred: ArrayLike, diabetes_type: int = 1) -> dict[str, Any]:
    """Parkes zone counts/percentages (``parkes_zone_A_count``, ``parkes_zone_A_pct`` ...)."""
    return zone_counts(parkes_zone(ref, pred, diabetes_type), "parkes_zone")


# --------------------------------------------------------------------------- #
# Hypo- / hyperglycaemia event detection
# --------------------------------------------------------------------------- #
def _binary_rates(truth: np.ndarray, call: np.ndarray) -> dict[str, float | int | None]:
    tp = int((truth & call).sum()); fn = int((truth & ~call).sum())
    fp = int((~truth & call).sum()); tn = int((~truth & ~call).sum())
    div = lambda a, b: float(a / b) if b else None
    return {"n_events": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)}


def event_metrics(ref: ArrayLike, pred: ArrayLike, hypo: float = 70.0, hyper: float = 180.0) -> dict[str, Any]:
    """Detection metrics for hypoglycaemia (``ref < hypo``) and hyperglycaemia (``ref > hyper``).

    Treating the regressor as an alarm (``pred < hypo`` / ``pred > hyper``)
    answers the clinically relevant question "would it have warned?".
    Sensitivity/specificity/PPV/NPV are ``None`` when undefined (no events in
    the cohort, which is the case for euglycaemic volunteers).  Thresholds
    follow the ADA consensus (70 and 180 mg/dL).
    """
    r, p = _pair(ref, pred)
    r, p = r.astype(np.float64), p.astype(np.float64)
    return {"hypo_threshold": float(hypo), "hyper_threshold": float(hyper),
            "hypo": _binary_rates(r < hypo, p < hypo),
            "hyper": _binary_rates(r > hyper, p > hyper)}


# --------------------------------------------------------------------------- #
# Record level and summary
# --------------------------------------------------------------------------- #
def record_level(pred: ArrayLike, ws: "WindowSet", drop_nan: bool = True) -> pd.DataFrame:
    """Average window predictions per record.

    Returns one row per ``(record, subject)`` with columns ``record, subject,
    y, pred, n_windows``; ``y`` is the (single) label of the record, ``pred``
    the mean window prediction.  Windows whose prediction is ``NaN`` (not in
    any test fold, e.g. ``--quick`` runs) are dropped first when ``drop_nan``,
    reproducing ``run_subject_cv``'s ``keep = ~isnan(pred)`` path.  Row order
    is the sorted ``(record, subject)`` order of ``DataFrame.groupby``, as in
    the legacy ``recording_level``.
    """
    pred = np.asarray(pred).reshape(-1)
    if len(pred) != len(ws):
        raise ValueError(f"pred has {len(pred)} entries but the WindowSet has {len(ws)} windows")
    record, subject, y = np.asarray(ws.record), np.asarray(ws.subject), np.asarray(ws.y)
    if drop_nan:
        keep = ~np.isnan(pred.astype(np.float64))
        if not keep.all():
            log.debug("record_level: dropping %d window(s) with NaN prediction", int((~keep).sum()))
        pred, record, subject, y = pred[keep], record[keep], subject[keep], y[keep]
    df = pd.DataFrame({"record": record, "subject": subject, "y": y, "pred": pred})
    return df.groupby(["record", "subject"], as_index=False).agg(
        y=("y", "first"), pred=("pred", "mean"), n_windows=("pred", "size"))


def summarize(df: pd.DataFrame, diabetes_type: int = 1, events: bool = True) -> dict[str, Any]:
    """Record-level metric dictionary for a ``record_level`` table.

    Keys (in order): ``n, mae, rmse, mard_pct, r2, pearson_r,
    within_subject_r_median, zone_A_pct..zone_E_pct`` (Clarke),
    ``parkes_zone_A_pct..parkes_zone_E_pct`` (consensus grid for
    ``diabetes_type``), ``bias``, ``n_subjects`` and, when ``events``,
    ``hypo``/``hyper`` detection dicts (:func:`event_metrics`).  The first
    twelve keys are computed exactly as ``run_subject_cv.metrics`` (same
    dtypes: no up-casting of float32 columns), so legacy summaries are
    reproduced bit for bit; the remaining keys are additive and ignored by
    the legacy macro generator.

    Degenerate input (fewer than two records, or zero reference variance)
    returns ``{"n", "mae", "error": "degenerate"}`` instead of raising.
    """
    y, p = df.y.to_numpy(), df.pred.to_numpy()
    if len(y) < 2 or y.std() == 0:
        return {"n": int(len(y)), "mae": float(np.abs(p - y).mean()) if len(y) else None, "error": DEGENERATE}
    e = p - y
    ss_res, ss_tot = float((e ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    cz = clarke_summary(y.astype(np.float32), p.astype(np.float32))
    pz = parkes_summary(y.astype(np.float32), p.astype(np.float32), diabetes_type)
    with np.errstate(invalid="ignore", divide="ignore"):
        within = within_subject_r(df).to_numpy()
        r = float(np.corrcoef(y, p)[0, 1])
        mard = float((np.abs(e) / y).mean() * 100)
    out: dict[str, Any] = {
        "n": int(len(df)), "mae": float(np.abs(e).mean()), "rmse": float(math.sqrt((e ** 2).mean())),
        "mard_pct": mard, "r2": 1 - ss_res / ss_tot, "pearson_r": r,
        "within_subject_r_median": float(np.nanmedian(within)) if np.isfinite(within).any() else None,
        **{f"zone_{z}_pct": cz[f"zone_{z}_pct"] for z in ZONES},
        **{f"parkes_zone_{z}_pct": pz[f"parkes_zone_{z}_pct"] for z in ZONES},
        "bias": float(e.mean()), "n_subjects": int(df.subject.nunique()),
    }
    if events:
        out.update(event_metrics(y, p))
    return out


def per_subject(df: pd.DataFrame, min_records: int = 6, **kw: Any) -> dict[str, dict[str, Any]]:
    """``{subject: summarize(rows of that subject)}`` for subjects with at least ``min_records`` records.

    ``min_records=6`` reproduces ``run_bigideas_cv`` (``len(g) > 5``).
    """
    return {str(s): summarize(g, **kw) for s, g in df.groupby("subject") if len(g) >= min_records}


def pool_seeds(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge record-level tables from several seeds into one (mean prediction per record).

    Reproduces ``bootstrap_ci.load``: subjects are zero-padded to three
    characters (CSV round-trips turn ``"01"`` into ``1``), then
    ``pred`` is averaged and ``y`` taken from the first row per record.
    """
    df = pd.concat(list(frames), ignore_index=True)
    df["subject"] = df.subject.astype(str).str.zfill(3)
    return df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))


def aggregate_seeds(runs: Sequence[Mapping[str, Any]], keys: Sequence[str] | None = None) -> dict[str, Any]:
    """Mean / std / n over seeds for every numeric metric.

    Returns ``{key: {"mean", "std", "n", "values"}}`` with the population std
    (``ddof=0``), matching the ``mean\\ensuremath{\\pm}std`` of
    ``make_results_macros``.  Non-finite or missing values are skipped.
    """
    runs = [r for r in runs if "error" not in r]
    if not runs:
        return {}
    if keys is None:
        keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [float(r[k]) for r in runs if r.get(k) is not None and np.isfinite(r.get(k))]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals), "values": vals}
    return out


# --------------------------------------------------------------------------- #
# Cluster bootstrap
# --------------------------------------------------------------------------- #
def _merge_with_null(df_model: pd.DataFrame, df_null: pd.DataFrame) -> pd.DataFrame:
    m = df_model.merge(df_null, on=["record", "subject"], suffixes=("", "_null"))
    if m.empty:
        raise ValueError("model and null tables share no (record, subject) rows")
    return m


def cluster_bootstrap(df_model: pd.DataFrame, df_null: pd.DataFrame, B: int = 5000, seed: int = 0,
                      rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Cluster-bootstrap (resampling subjects) 95 % CI for MAE and for the paired MAE difference vs a null model.

    Identical to ``bootstrap_ci.cluster_bootstrap``: the two record-level
    tables are inner-joined on ``(record, subject)``; in each of ``B``
    replicates ``n_subjects`` subjects are drawn with replacement
    (``rng.choice(subjects, n, replace=True)``, subjects in first-appearance
    order of the merged table), the per-record absolute errors of the drawn
    subjects are concatenated and averaged for both models, and the paired
    difference ``MAE_model - MAE_null`` is taken on the *same* draw.  The
    reported CI is the 2.5 / 97.5 percentile of the replicates and
    ``p_better_than_null`` is the fraction of replicates in which the model
    beats the null.  A difference CI excluding zero is "significant" at the
    5 % level (two-sided).

    ``rng`` lets a caller thread one generator through several models, which
    is what the legacy script did; otherwise ``np.random.default_rng(seed)``
    is used.  Returns ``error="degenerate"`` for fewer than two subjects.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    m = _merge_with_null(df_model, df_null)
    subs = m.subject.unique()
    if len(subs) < 2:
        return {"error": DEGENERATE, "n_subjects": int(len(subs)), "n_records": int(len(m)),
                "mae": float(np.abs(m.pred - m.y).mean())}
    by = {s: g for s, g in m.groupby("subject")}
    e_m = {s: np.abs(g.pred - g.y).to_numpy() for s, g in by.items()}
    e_n = {s: np.abs(g.pred_null - g.y).to_numpy() for s, g in by.items()}
    maes, diffs = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.choice(subs, len(subs), replace=True)
        em = np.concatenate([e_m[s] for s in pick]); en = np.concatenate([e_n[s] for s in pick])
        maes[b] = em.mean(); diffs[b] = em.mean() - en.mean()
    mae_ = float(np.abs(m.pred - m.y).mean()); diff = mae_ - float(np.abs(m.pred_null - m.y).mean())
    return dict(mae=mae_, mae_ci=[float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
                diff_vs_null=diff, diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                p_better_than_null=float((diffs < 0).mean()), n_subjects=int(len(subs)), n_records=int(len(m)))


def cluster_bootstrap_many(tables: Mapping[str, pd.DataFrame], df_null: pd.DataFrame, B: int = 5000,
                           seed: int = 0) -> dict[str, dict[str, Any]]:
    """:func:`cluster_bootstrap` for several models with ONE generator threaded through them in order.

    This is how ``bootstrap_ci.main`` produced ``results/bootstrap_ci.json``:
    ``np.random.default_rng(seed)`` is created once and consumed by the
    models in sequence, so the replicate draws (and hence the confidence
    intervals) of the *k*-th model depend on the models bootstrapped before
    it.  Reproducing a stored table therefore needs the same ``tables`` in
    the same insertion order; bootstrapping one model with a fresh generator
    gives the same point estimates but a different CI.  Returns
    ``{name: cluster_bootstrap(table, df_null)}`` in the order of ``tables``.
    """
    rng = np.random.default_rng(seed)
    return {str(name): cluster_bootstrap(df, df_null, B=B, rng=rng) for name, df in tables.items()}


def _subject_blocks(df: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    subs = df.subject.unique()
    rows = {s: np.flatnonzero(df.subject.to_numpy() == s) for s in subs}
    return subs, [rows[s] for s in subs]


def cluster_bootstrap_metrics(df: pd.DataFrame, B: int = 2000, seed: int = 0,
                              metrics: Sequence[str] = ("mae", "rmse", "mard_pct", "pearson_r",
                                                        "within_subject_r_median", "zone_A_pct",
                                                        "parkes_zone_A_pct"),
                              alpha: float = 0.05, min_subjects: int = 2) -> dict[str, Any]:
    """Subject-resampled percentile CIs for every metric of :func:`summarize`.

    Each replicate draws subjects with replacement and re-runs
    :func:`summarize` on their pooled records.  Replicates that are degenerate
    (e.g. one subject with a single glucose value) are skipped and counted in
    ``n_degenerate``.  Returns ``{metric: {"point", "lo", "hi", "n_boot"}}``
    plus ``n_subjects`` / ``n_degenerate``; ``error="degenerate"`` with fewer
    than ``min_subjects`` subjects.
    """
    rng = np.random.default_rng(seed)
    subs, blocks = _subject_blocks(df)
    point = summarize(df, events=False)
    if len(subs) < min_subjects or "error" in point:
        return {"error": DEGENERATE, "n_subjects": int(len(subs))}
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    n_deg = 0
    for _ in range(B):
        pick = rng.integers(0, len(subs), len(subs))
        rep = summarize(df.iloc[np.concatenate([blocks[i] for i in pick])], events=False)
        if "error" in rep:
            n_deg += 1
            continue
        for k in metrics:
            v = rep.get(k)
            draws[k].append(float("nan") if v is None else float(v))
    out: dict[str, Any] = {"n_subjects": int(len(subs)), "n_degenerate": n_deg, "B": int(B), "alpha": alpha}
    for k in metrics:
        v = np.asarray(draws[k], dtype=float)
        v = v[np.isfinite(v)]
        pv = point.get(k)
        out[k] = {"point": None if pv is None else float(pv),
                  "lo": float(np.percentile(v, 100 * alpha / 2)) if v.size else None,
                  "hi": float(np.percentile(v, 100 * (1 - alpha / 2))) if v.size else None,
                  "n_boot": int(v.size)}
    return out


def paired_cluster_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame,
                             metric: str | Callable[[np.ndarray, np.ndarray], float] = "mae",
                             B: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Paired subject-bootstrap CI of ``metric(A) - metric(B)`` for two models on the same records.

    Generalises :func:`cluster_bootstrap` to any record-level metric
    (``"mae"``, ``"rmse"``, ``"mard_pct"``, ``"pearson_r"`` or a callable
    ``f(y, pred) -> float``).  Both models are evaluated on the same resampled
    subjects, so the difference respects pairing and within-subject
    correlation.  ``p_a_better`` is the fraction of replicates where A has
    the lower value (use ``1 - p`` for "higher is better" metrics such as r).
    """
    fn = {"mae": mae, "rmse": rmse, "mard_pct": mard_pct, "pearson_r": pearson_r, "r2": r2}[metric] \
        if isinstance(metric, str) else metric
    m = df_a.merge(df_b, on=["record", "subject"], suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("the two tables share no (record, subject) rows")
    subs, blocks = _subject_blocks(m)
    if len(subs) < 2:
        return {"error": DEGENERATE, "n_subjects": int(len(subs))}
    y, pa, pb = m.y_a.to_numpy(), m.pred_a.to_numpy(), m.pred_b.to_numpy()
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        for b in range(B):
            idx = np.concatenate([blocks[i] for i in rng.integers(0, len(subs), len(subs))])
            d[b] = fn(y[idx], pa[idx]) - fn(y[idx], pb[idx])
        point = fn(y, pa) - fn(y, pb)
    d = d[np.isfinite(d)]
    return {"metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
            "diff": float(point), "diff_ci": [float(np.percentile(d, 100 * alpha / 2)),
                                              float(np.percentile(d, 100 * (1 - alpha / 2)))],
            "p_a_better": float((d < 0).mean()) if d.size else None,
            "n_subjects": int(len(subs)), "n_records": int(len(m)), "n_boot": int(d.size)}
