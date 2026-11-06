"""
Numeric parity of ``ppgbench`` with the legacy reference scripts on the REAL
Mendeley recordings and the stored result files in ``results/``.

The framework specification (``docs/framework_spec.md``) makes the legacy
scripts the reference behaviour: the package must reproduce their numbers
(same preprocessing, same splits, same metrics).  The fast unit tests check
this on synthetic fixtures; this module checks it end to end on the data and
on the numbers that the paper actually reports.

Checks (numbering follows the review brief)
-------------------------------------------
1. ``load_dataset("mendeley")`` == ``ppg_data.load_mendeley()``:
   ``X`` (allclose and bit-exact), ``y``, ``subject``, ``record``, ``meta``.
2. ``ppgbench.core.views.make_views`` == ``ppg_data.make_views`` (``x1``, ``x2``, ``x3``).
3. ``ppgbench.core.features.handcrafted_features`` == ``ppg_data.handcrafted_features``.
4. ``make_folds(ws, "subject_loso")`` == the ``LeaveOneGroupOut`` folds of ``run_subject_cv``.
5. ``mean`` / ``ridge`` (and ``demo``, ``ridge+demo``) under ``subject_loso``
   reproduce ``run_subject_cv.predict_classical`` window for window, the
   stored out-of-fold CSVs, and the record-level MAE in
   ``results/subject_cv_summary.json`` (mean 14.71, ridge 14.62).
6. ``metrics.summarize`` on ``results/subject_cv_fusion_seed0.csv`` reproduces
   the stored ``fusion_seed0`` metrics.
7. ``metrics.cluster_bootstrap`` reproduces ``results/bootstrap_ci.json`` for
   ``mendeley/1d`` (and every other ``mendeley/*`` entry) when one seed-0
   generator is threaded through the models in the legacy order, exactly as
   ``bootstrap_ci.main`` did.

Each test prints one ``PARITY <n> ... PASS|FAIL`` line carrying the numbers it
compared, and a summary table is printed when the module finishes, so
``pytest -s`` doubles as the parity report.

Cost and gating
---------------
The tests read the 67 raw recordings twice (legacy and ppgbench), compute the
three views, run the classical LOSO baselines and 7 x 5000 bootstrap
replicates: ~30-60 s in total.  **No neural network is trained.**  They are
marked ``slow`` and skipped unless ``PPGBENCH_SLOW=1``::

    PPGBENCH_SLOW=1 ~/.venvs/ppgfusion/bin/python -m pytest tests/ppgbench/test_parity.py -s -v

They also skip (not fail) when the raw data or a stored result file is absent,
so a clone without the datasets still has a green test suite.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
MENDELEY_ROOT = ROOT / "Archive_Dataset" / "archive (2)" / "PPG_Dataset"

pytestmark = pytest.mark.slow

#: Metric keys written by ``run_subject_cv.metrics`` (the legacy summary), in order.
LEGACY_METRIC_KEYS: tuple[str, ...] = (
    "n", "mae", "rmse", "mard_pct", "r2", "pearson_r", "within_subject_r_median",
    "zone_A_pct", "zone_B_pct", "zone_C_pct", "zone_D_pct", "zone_E_pct",
)
#: Keys of one ``bootstrap_ci.cluster_bootstrap`` entry.
BOOTSTRAP_KEYS: tuple[str, ...] = (
    "mae", "mae_ci", "diff_vs_null", "diff_ci", "p_better_than_null", "n_subjects", "n_records",
)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_REPORT: list[tuple[str, bool, str]] = []


def _report(check: str, ok: bool, detail: str) -> None:
    """Record and print one parity line (``PARITY <check>: PASS|FAIL | numbers``)."""
    _REPORT.append((check, ok, detail))
    print(f"\nPARITY {check}: {'PASS' if ok else 'FAIL'} | {detail}")


@pytest.fixture(scope="module", autouse=True)
def _parity_summary():
    """Print the summary table once every test in this module has run."""
    yield
    if not _REPORT:
        return
    width = max(len(c) for c, _, _ in _REPORT)
    print("\n" + "=" * 78 + "\nppgbench numeric parity summary\n" + "-" * 78)
    for check, ok, detail in _REPORT:
        print(f"{check:<{width}}  {'PASS' if ok else 'FAIL'}  {detail}")
    n_fail = sum(not ok for _, ok, _ in _REPORT)
    print("-" * 78 + f"\n{len(_REPORT) - n_fail}/{len(_REPORT)} checks passed\n" + "=" * 78)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()) if np.size(a) else 0.0


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")
    return path


# --------------------------------------------------------------------------- #
# Fixtures: the legacy reference and the ppgbench objects (module scope: loaded once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def legacy():
    """``ppg_data`` - the legacy loaders, views and features."""
    if not MENDELEY_ROOT.exists():
        pytest.skip(f"raw Mendeley data not present under {MENDELEY_ROOT}")
    return pytest.importorskip("ppg_data")


@pytest.fixture(scope="module")
def legacy_cv():
    """``run_subject_cv`` - the legacy LOSO harness (classical predictor, record-level metrics)."""
    return pytest.importorskip("run_subject_cv")


@pytest.fixture(scope="module")
def ws_legacy(legacy):
    return legacy.load_mendeley()


@pytest.fixture(scope="module")
def ws_bench(legacy):
    from ppgbench.data import load_dataset
    return load_dataset("mendeley")


@pytest.fixture(scope="module")
def folds_legacy(ws_legacy):
    """Exactly the splitter call of ``run_subject_cv.main`` with ``--folds loso``."""
    from sklearn.model_selection import LeaveOneGroupOut
    return list(LeaveOneGroupOut().split(ws_legacy.X, ws_legacy.y, ws_legacy.subject))


@pytest.fixture(scope="module")
def folds_bench(ws_bench):
    from ppgbench.protocols import make_folds
    return make_folds(ws_bench, "subject_loso")


@pytest.fixture(scope="module")
def feats_legacy(legacy, ws_legacy):
    """``(hand, demo)`` built exactly as in ``run_subject_cv.main``."""
    hand = legacy.handcrafted_features(ws_legacy.X)
    m = ws_legacy.meta.set_index("record")
    demo = m.loc[ws_legacy.record, ["age", "height", "weight"]].to_numpy(np.float32)
    demo = np.hstack([demo, (m.loc[ws_legacy.record, "gender"] == "Male").to_numpy()[:, None]])
    return hand, demo


@pytest.fixture(scope="module")
def feats_bench(ws_bench):
    from ppgbench.core.features import feature_bundle
    return feature_bundle(ws_bench)


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return json.loads(_need(RESULTS / "subject_cv_summary.json", "legacy LOSO summary").read_text())


# --------------------------------------------------------------------------- #
# 1. dataset loader
# --------------------------------------------------------------------------- #
def test_1_load_mendeley_matches_legacy(ws_legacy, ws_bench):
    """``load_dataset('mendeley')`` reproduces ``ppg_data.load_mendeley`` array for array."""
    same_shape = ws_bench.X.shape == ws_legacy.X.shape
    dx = _max_abs(ws_bench.X, ws_legacy.X) if same_shape else float("inf")
    x_close = same_shape and np.allclose(ws_bench.X, ws_legacy.X, rtol=0.0, atol=1e-6)
    x_exact = same_shape and np.array_equal(ws_bench.X, ws_legacy.X)
    y_eq = np.array_equal(ws_bench.y, ws_legacy.y)
    s_eq = np.array_equal(ws_bench.subject, ws_legacy.subject)
    r_eq = np.array_equal(ws_bench.record, ws_legacy.record)
    meta_cols = ("record", "subject", "glucose", "gender", "age", "height", "weight", "raw_std", "n_windows")
    meta_eq = (len(ws_bench.meta) == len(ws_legacy.meta)
               and all(np.array_equal(ws_bench.meta[c].to_numpy(), ws_legacy.meta[c].to_numpy()) for c in meta_cols))
    dtypes_ok = ws_bench.X.dtype == np.float32 and ws_bench.y.dtype == np.float32 and ws_bench.time is None
    ok = x_close and y_eq and s_eq and r_eq and meta_eq and dtypes_ok
    _report("1 load_dataset(mendeley) vs ppg_data.load_mendeley", ok,
            f"X {ws_bench.X.shape} vs {ws_legacy.X.shape} max|dX|={dx:.3g} exact={x_exact}; "
            f"y={y_eq} subject={s_eq} record={r_eq} meta[{len(meta_cols)} cols]={meta_eq}; "
            f"{ws_bench.n_records} records / {ws_bench.n_subjects} subjects / {len(ws_bench)} windows; time=None={ws_bench.time is None}")
    assert same_shape, f"window count differs: {ws_bench.X.shape} vs {ws_legacy.X.shape}"
    assert x_close, f"X differs by up to {dx}"
    assert y_eq and s_eq and r_eq, "y / subject / record arrays differ"
    assert meta_eq, "record-level meta differs"
    assert dtypes_ok


# --------------------------------------------------------------------------- #
# 2. model views
# --------------------------------------------------------------------------- #
def test_2_make_views_matches_legacy(legacy, ws_legacy):
    """``make_views`` (CWT scalogram, Takens cloud) is bit-identical on the real windows."""
    import torch
    from ppgbench.core.views import make_views

    X = ws_legacy.X                               # same input for both: isolates the transform itself
    theirs, ours = legacy.make_views(X), make_views(X)
    parts = []
    ok = set(theirs) == set(ours) == {"x1", "x2", "x3"}
    for k in ("x1", "x2", "x3"):
        same = tuple(theirs[k].shape) == tuple(ours[k].shape) and theirs[k].dtype == ours[k].dtype == torch.float32
        d = float((theirs[k] - ours[k]).abs().max()) if same else float("inf")
        eq = same and torch.equal(theirs[k], ours[k])
        parts.append(f"{k}{tuple(ours[k].shape)} max|d|={d:.3g} exact={eq}")
        ok = ok and same and bool(torch.allclose(theirs[k], ours[k], rtol=0.0, atol=1e-6))
    _report("2 make_views vs ppg_data.make_views", ok, "; ".join(parts))
    assert ok, "; ".join(parts)


# --------------------------------------------------------------------------- #
# 3. hand-crafted features
# --------------------------------------------------------------------------- #
def test_3_handcrafted_features_match_legacy(legacy, ws_legacy):
    """The nine classical descriptors are identical column for column."""
    from ppgbench.core.features import FEATURE_NAMES, handcrafted_features

    theirs, ours = legacy.handcrafted_features(ws_legacy.X), handcrafted_features(ws_legacy.X)
    same = theirs.shape == ours.shape == (len(ws_legacy), 9) and ours.dtype == np.float32
    per_col = np.abs(theirs.astype(np.float64) - ours.astype(np.float64)).max(0) if same else None
    exact = same and np.array_equal(theirs, ours)
    ok = same and np.allclose(theirs, ours, rtol=1e-6, atol=1e-6)
    cols = ", ".join(f"{n}={d:.2g}" for n, d in zip(FEATURE_NAMES, per_col)) if same else "shape mismatch"
    _report("3 handcrafted_features vs ppg_data.handcrafted_features", ok,
            f"shape {ours.shape} exact={exact}; max|d| per column: {cols}")
    assert ok, cols


# --------------------------------------------------------------------------- #
# 4. LOSO folds
# --------------------------------------------------------------------------- #
def test_4_subject_loso_folds_match_run_subject_cv(ws_legacy, folds_legacy, folds_bench):
    """``make_folds(ws, 'subject_loso')`` == ``LeaveOneGroupOut().split(X, y, subject)`` fold for fold."""
    n_eq = len(folds_legacy) == len(folds_bench)
    pair_eq = n_eq and all(np.array_equal(a, c) and np.array_equal(b, d)
                           for (a, b), (c, d) in zip(folds_legacy, folds_bench))
    held_out = [np.unique(ws_legacy.subject[te]).tolist() for _, te in folds_bench]
    one_subject_each = all(len(h) == 1 for h in held_out)
    covers_all = sorted(h[0] for h in held_out) == sorted(np.unique(ws_legacy.subject).tolist()) if one_subject_each else False
    sizes = [int(len(te)) for _, te in folds_bench]
    ok = n_eq and pair_eq and one_subject_each and covers_all
    _report("4 make_folds(subject_loso) vs LeaveOneGroupOut", ok,
            f"{len(folds_bench)} folds (legacy {len(folds_legacy)}), all train/test index arrays equal={pair_eq}; "
            f"one held-out subject per fold={one_subject_each}, every subject held out once={covers_all}; "
            f"test sizes min/max={min(sizes)}/{max(sizes)} windows")
    assert ok


# --------------------------------------------------------------------------- #
# 5. classical baselines under LOSO
# --------------------------------------------------------------------------- #
def _loso_predictions(name: str, ws_bench, ws_legacy, folds_bench, feats_bench, feats_legacy, legacy_cv
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold window predictions from ppgbench and from ``run_subject_cv.predict_classical``."""
    from ppgbench.models import build_model

    hand, demo = feats_legacy
    ours = np.full(len(ws_bench), np.nan, np.float32)
    theirs = np.full(len(ws_legacy), np.nan, np.float32)
    for tr, te in folds_bench:
        ours[te] = build_model(name).fit(ws_bench, tr, feats_bench).predict(ws_bench, te, feats_bench)
        theirs[te] = legacy_cv.predict_classical(name, hand, demo, ws_legacy.y, tr, te)
    return ours, theirs


@pytest.mark.parametrize("name, expected_mae", [("mean", 14.71), ("ridge", 14.62), ("demo", 15.57), ("ridge+demo", 15.23)])
def test_5_classical_loso_mae_matches_summary(name, expected_mae, ws_bench, ws_legacy, folds_bench, feats_bench,
                                              feats_legacy, legacy_cv, summary):
    """Record-level LOSO MAE of the classical models equals ``subject_cv_summary.json`` (and the stored CSV)."""
    from ppgbench.metrics import record_level, summarize

    key = f"{name}_seed0"
    if key not in summary:
        pytest.skip(f"{key} not in subject_cv_summary.json")
    stored = summary[key]
    csv_path = _need(RESULTS / f"subject_cv_{name}_seed0.csv", f"stored LOSO predictions for {name}")

    ours, theirs = _loso_predictions(name, ws_bench, ws_legacy, folds_bench, feats_bench, feats_legacy, legacy_cv)
    assert not np.isnan(ours).any() and not np.isnan(theirs).any(), "every window must be predicted once"
    d_win = _max_abs(ours, theirs)
    win_exact = np.array_equal(ours, theirs)

    df = record_level(ours, ws_bench)
    m = summarize(df)
    m_legacy = legacy_cv.metrics(legacy_cv.recording_level(theirs, ws_legacy))
    csv = pd.read_csv(csv_path)
    merged = df.merge(csv, on="record", suffixes=("", "_csv"))
    d_csv = _max_abs(merged.pred, merged.pred_csv) if len(merged) == len(df) == len(csv) else float("inf")
    d_y = _max_abs(merged.y, merged.y_csv) if len(merged) else float("inf")

    mae_2dp = round(m["mae"], 2)
    d_mae = abs(m["mae"] - stored["mae"])
    metric_diffs = {k: abs(m[k] - stored[k]) for k in LEGACY_METRIC_KEYS
                    if isinstance(stored.get(k), (int, float)) and m.get(k) is not None}
    ok = (mae_2dp == expected_mae == round(stored["mae"], 2) and d_mae < 1e-6
          and np.allclose(ours, theirs, rtol=0.0, atol=1e-4) and d_csv < 1e-4 and d_y == 0.0
          and all(v < 1e-6 for v in metric_diffs.values()) and m["n"] == stored["n"] == 67)
    _report(f"5 {name:<10s} LOSO MAE vs subject_cv_summary.json", ok,
            f"MAE ppgbench={m['mae']:.6f} legacy-fn={m_legacy['mae']:.6f} stored={stored['mae']:.6f} "
            f"(2 dp {mae_2dp:.2f} vs expected {expected_mae:.2f}); n={m['n']}; "
            f"window max|ours-legacy|={d_win:.3g} exact={win_exact}; record max|ours-CSV|={d_csv:.3g}; "
            f"max metric |d| vs stored={max(metric_diffs.values()):.3g} over {len(metric_diffs)} keys")
    assert mae_2dp == expected_mae, f"{name}: MAE {m['mae']:.4f} != {expected_mae}"
    assert d_mae < 1e-6, f"{name}: MAE {m['mae']!r} vs stored {stored['mae']!r}"
    assert np.allclose(ours, theirs, rtol=0.0, atol=1e-4), f"{name}: window predictions differ by {d_win}"
    assert d_csv < 1e-4 and d_y == 0.0, f"{name}: record-level predictions differ from the stored CSV by {d_csv}"
    assert all(v < 1e-6 for v in metric_diffs.values()), metric_diffs


def test_5b_classical_predictions_bit_exact(ws_bench, ws_legacy, folds_bench, feats_bench, feats_legacy, legacy_cv):
    """Stronger than the MAE check: the float32 feature pipeline reproduces the legacy predictions bit for bit.

    ``assemble_features`` keeps the float32 dtype of the hand-crafted / demographic
    blocks, so ``StandardScaler`` + ``RidgeCV`` run in the same precision as
    ``run_subject_cv.predict_classical`` and the out-of-fold predictions are
    identical rather than one float32 ulp apart.
    """
    parts, ok = [], True
    for name in ("mean", "ridge", "demo", "ridge+demo"):
        ours, theirs = _loso_predictions(name, ws_bench, ws_legacy, folds_bench, feats_bench, feats_legacy, legacy_cv)
        eq = np.array_equal(ours, theirs)
        parts.append(f"{name}: exact={eq} max|d|={_max_abs(ours, theirs):.3g}")
        ok = ok and eq
    _report("5b classical window predictions bit-exact vs predict_classical", ok, "; ".join(parts))
    assert ok, "; ".join(parts)


# --------------------------------------------------------------------------- #
# 6. summarize on a stored record-level table
# --------------------------------------------------------------------------- #
def test_6_summarize_reproduces_stored_fusion_metrics(summary, legacy_cv):
    """``summarize(subject_cv_fusion_seed0.csv)`` equals the stored ``fusion_seed0`` entry.

    The legacy ``metrics`` ran on an in-memory table whose ``y`` column was
    float32 (``ws.y``) while ``pred`` was float64 (pandas up-casts the group
    mean).  A CSV round-trip turns ``y`` into float64, which perturbs ``r2``
    in the 8th decimal through ``((y - y.mean()) ** 2).sum()``.  The test
    therefore (i) restores the legacy dtype and demands exact equality and
    (ii) checks the raw CSV to 1e-6 and shows that the legacy function itself
    produces the same perturbation, so the drift is a round-trip artefact, not
    a ppgbench defect.
    """
    from ppgbench.metrics import summarize

    stored = summary.get("fusion_seed0") or pytest.skip("fusion_seed0 not in subject_cv_summary.json")
    csv = pd.read_csv(_need(RESULTS / "subject_cv_fusion_seed0.csv", "stored fusion predictions"))
    as_legacy = csv.assign(y=csv.y.astype(np.float32))

    def diffs(m: dict[str, Any]) -> dict[str, float]:
        out = {}
        for k in LEGACY_METRIC_KEYS:
            a, b = m.get(k), stored.get(k)
            out[k] = 0.0 if a is None and b is None else float("inf") if (a is None) != (b is None) else abs(a - b)
        return out

    d_exact = diffs(summarize(as_legacy))
    d_raw = diffs(summarize(csv))
    d_raw_legacy_fn = diffs(legacy_cv.metrics(csv))
    extra = {k for k in summarize(csv) if k not in LEGACY_METRIC_KEYS}
    ok = (max(d_exact.values()) == 0.0 and max(d_raw.values()) < 1e-6
          and all(d_raw[k] == d_raw_legacy_fn[k] for k in LEGACY_METRIC_KEYS))
    _report("6 summarize(subject_cv_fusion_seed0.csv) vs stored fusion_seed0", ok,
            f"n={stored['n']} mae={stored['mae']:.6f} r2={stored['r2']:.6f} zoneA={stored['zone_A_pct']:.2f}%; "
            f"max|d| with legacy dtypes={max(d_exact.values()):.3g} (all {len(d_exact)} keys), "
            f"raw CSV={max(d_raw.values()):.3g} (legacy metrics() on the same CSV: {max(d_raw_legacy_fn.values()):.3g}); "
            f"{len(extra)} additive keys ignored")
    assert max(d_exact.values()) == 0.0, d_exact
    assert max(d_raw.values()) < 1e-6, d_raw
    assert d_raw == d_raw_legacy_fn, "ppgbench and the legacy function disagree on the raw CSV"


# --------------------------------------------------------------------------- #
# 7. cluster bootstrap
# --------------------------------------------------------------------------- #
def _load_legacy_table(pattern: str) -> pd.DataFrame | None:
    """``bootstrap_ci.load``: concatenate the seed files of one model and average per record."""
    from ppgbench.metrics import pool_seeds
    files = sorted(glob.glob(str(RESULTS / pattern)))
    return pool_seeds([pd.read_csv(f) for f in files]) if files else None


def test_7_cluster_bootstrap_reproduces_bootstrap_ci_json(legacy_cv):
    """``cluster_bootstrap`` reproduces ``bootstrap_ci.json`` for ``mendeley/1d`` (and all mendeley entries).

    ``bootstrap_ci.main`` seeds ONE ``default_rng(0)`` and threads it through
    the models in order (ridge, demo, ridge+demo, 1d, ...), so the draws of
    ``1d`` depend on the three models bootstrapped before it.  The test
    replays that order (the key order of the JSON) with
    :func:`ppgbench.metrics.cluster_bootstrap_many` and demands exact equality
    of every value.  A *fresh* seed-0 generator reproduces the point MAE but
    necessarily yields a different CI; that is reported, not asserted.
    """
    from ppgbench.metrics import cluster_bootstrap, cluster_bootstrap_many

    ref = json.loads(_need(RESULTS / "bootstrap_ci.json", "stored bootstrap CIs").read_text())
    keys = [k for k in ref if k.startswith("mendeley/")]
    if "mendeley/1d" not in keys:
        pytest.skip("bootstrap_ci.json has no mendeley/1d entry")
    null = _load_legacy_table("subject_cv_mean_seed0.csv")
    if null is None:
        pytest.skip("subject_cv_mean_seed0.csv missing")
    tables = {k: _load_legacy_table(f"subject_cv_{k.split('/', 1)[1]}_seed*.csv") for k in keys}
    missing = [k for k, t in tables.items() if t is None]
    if missing:
        pytest.skip(f"stored predictions missing for {missing}")

    out = cluster_bootstrap_many(tables, null, B=5000, seed=0)
    exact = {k: out[k] == ref[k] for k in keys}
    worst = {k: max(_max_abs(np.ravel(out[k][f]), np.ravel(ref[k][f])) for f in BOOTSTRAP_KEYS) for k in keys}
    one = out["mendeley/1d"]
    fresh = cluster_bootstrap(tables["mendeley/1d"], null, B=5000, seed=0)
    ok = all(exact.values())
    _report("7 cluster_bootstrap vs bootstrap_ci.json (mendeley/1d)", exact["mendeley/1d"] and ok,
            f"1d: mae={one['mae']:.6f} ci=[{one['mae_ci'][0]:.4f}, {one['mae_ci'][1]:.4f}] "
            f"diff={one['diff_vs_null']:+.4f} ci=[{one['diff_ci'][0]:+.4f}, {one['diff_ci'][1]:+.4f}] "
            f"p={one['p_better_than_null']:.4f} n_subj={one['n_subjects']} n_rec={one['n_records']} "
            f"exact={exact['mendeley/1d']}; all {len(keys)} mendeley entries exact={ok} "
            f"(worst |d| {max(worst.values()):.3g}); fresh default_rng(0) for 1d alone: mae={fresh['mae']:.6f} "
            f"(same point estimate={fresh['mae'] == ref['mendeley/1d']['mae']}), "
            f"ci=[{fresh['mae_ci'][0]:.4f}, {fresh['mae_ci'][1]:.4f}] (CI depends on generator position)")
    assert exact["mendeley/1d"], {f: (one[f], ref["mendeley/1d"][f]) for f in BOOTSTRAP_KEYS if one[f] != ref["mendeley/1d"][f]}
    assert ok, {k: worst[k] for k, e in exact.items() if not e}
    assert fresh["mae"] == ref["mendeley/1d"]["mae"]
    # the first entry uses an untouched generator, so a fresh seed also reproduces it exactly
    assert cluster_bootstrap(tables[keys[0]], null, B=5000, seed=0) == ref[keys[0]]


if __name__ == "__main__":  # pragma: no cover - convenience: `python tests/ppgbench/test_parity.py`
    os.environ.setdefault("PPGBENCH_SLOW", "1")
    sys.exit(pytest.main([__file__, "-s", "-v", "-p", "no:cacheprovider"]))
