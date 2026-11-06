"""
Tests for :mod:`ppgbench.metrics`.

Fast tests use small hand-built tables.  Parity tests against the legacy
scripts (``plot_clarke_error_grid``, ``run_subject_cv.metrics``,
``bootstrap_ci``) run whenever the legacy module / result files are present
and are skipped otherwise; they need no training and finish in seconds.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppgbench import metrics as M  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def table() -> pd.DataFrame:
    """Record-level table: 4 subjects x 5 records with a tracking signal plus noise."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(4):
        base = 90 + 15 * s
        for r in range(5):
            y = base + 6 * r
            rows.append({"record": f"{s:02d}_{r:04d}", "subject": f"{s:02d}", "y": float(y),
                         "pred": float(y + 4 * rng.standard_normal() + 2), "n_windows": 3})
    return pd.DataFrame(rows)


class _WS:
    """Minimal WindowSet stand-in (only the attributes record_level needs)."""

    def __init__(self, record, subject, y):
        self.record, self.subject, self.y = np.asarray(record), np.asarray(subject), np.asarray(y, np.float32)

    def __len__(self):
        return len(self.y)


# --------------------------------------------------------------------------- #
# Elementary metrics
# --------------------------------------------------------------------------- #
def test_elementary_metrics_known_values():
    y, p = np.array([100.0, 120.0, 140.0, 160.0]), np.array([110.0, 110.0, 150.0, 150.0])
    assert M.mae(y, p) == 10.0
    assert M.rmse(y, p) == 10.0
    assert M.bias(y, p) == 0.0
    assert np.isclose(M.mard_pct(y, p), np.mean([10 / 100, 10 / 120, 10 / 140, 10 / 160]) * 100)
    assert np.isclose(M.r2(y, p), 1 - 400 / 2000)
    assert np.isclose(M.pearson_r(y, p), np.corrcoef(y, p)[0, 1])
    assert np.isnan(M.pearson_r(y, np.full(4, 1.0)))
    assert np.isnan(M.r2(np.full(4, 1.0), p))


def test_within_subject_r(table):
    w = M.within_subject_r(table)
    assert set(w.index) == {"00", "01", "02", "03"} and np.isfinite(w).all()
    # a subject with constant reference contributes NaN, a tiny one is ignored
    t = table.copy()
    t.loc[t.subject == "00", "y"] = 100.0
    t = pd.concat([t, pd.DataFrame({"record": ["z"], "subject": ["zz"], "y": [1.0], "pred": [1.0], "n_windows": [1]})])
    w = M.within_subject_r(t)
    assert np.isnan(w["00"]) and np.isnan(w["zz"]) and np.isfinite(w[["01", "02", "03"]]).all()
    assert M.within_subject_r_median(t) == pytest.approx(np.nanmedian(w))
    assert M.within_subject_r_median(t[t.subject == "00"]) is None


# --------------------------------------------------------------------------- #
# Clarke
# --------------------------------------------------------------------------- #
def test_clarke_zone_textbook_points():
    r = np.array([100, 60, 100, 250, 100, 300, 50, 100, 120])
    p = np.array([110, 75, 230, 150, 60, 60, 250, 20, 235])
    #              A    A    C   D    D    E    E    D    C
    assert M.clarke_zone(r, p).tolist() == ["A", "A", "C", "D", "D", "E", "E", "D", "C"]
    assert M.clarke_zone([100], [125]).tolist() == ["B"]
    s = M.clarke_summary(r, p)
    assert s["total_points"] == 9 and s["zone_A_count"] == 2 and s["zone_A_pct"] == pytest.approx(200 / 9)
    assert sum(s[f"zone_{z}_count"] for z in M.ZONES) == 9


def test_clarke_matches_legacy_classifier_on_dense_grid():
    legacy = pytest.importorskip("plot_clarke_error_grid")
    g = np.arange(0, 451, 3.0)
    R, P = (a.ravel() for a in np.meshgrid(g, g))
    R32, P32 = R.astype(np.float32), P.astype(np.float32)
    mine = M.clarke_zone(R32, P32)
    ref = np.array([legacy._classify_point(float(r), float(p)) for r, p in zip(R32, P32)])
    assert (mine == ref).all()
    rng = np.random.default_rng(1)
    r = rng.uniform(20, 450, 3000).astype(np.float32)
    p = (r + rng.normal(0, 60, 3000)).astype(np.float32)
    assert M.clarke_summary(r, p) == legacy.summarize_clarke_zones(r, p)


# --------------------------------------------------------------------------- #
# Parkes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ref,pred,t1,t2", [
    (100, 100, "A", "A"),     # diagonal
    (100, 160, "B", "B"),     # above A-upper (126), below B-upper (180)
    (100, 190, "C", "B"),     # type 2 B-upper at x=100 is 197
    (100, 400, "D", "C"),
    (20, 300, "E", "E"),      # gross over-estimate of a hypo reference
    (300, 100, "C", "C"),     # under-estimate: right of B-lower (218), left of C-lower (414)
    (500, 50, "D", "D"),      # lower half extends to D only (no lower E)
    (60, 100, "C", "B"),
    (600, 600, "A", "A"),     # beyond 550: lines extrapolated
    (0, 50, "A", "A"),        # on the A-upper boundary -> inner zone
    (0, 50.001, "B", "B"),
])
def test_parkes_zone_known_points(ref, pred, t1, t2):
    assert M.parkes_zone([ref], [pred], 1)[0] == t1
    assert M.parkes_zone([ref], [pred], 2)[0] == t2


def test_parkes_vertices_are_nested_and_monotone():
    """Zones never improve when the prediction moves away from the reference, both grids."""
    rng = np.random.default_rng(0)
    for t in (1, 2):
        for z in ("upper", "lower"):
            v = M.PARKES_VERTICES[t][z]
            assert list(v) == (["A", "B", "C", "D"] if z == "upper" else ["A", "B", "C"])
        for r in rng.uniform(1, 549, 100):
            for p in (np.linspace(r, 700, 300), np.linspace(r, -50, 300)):
                idx = np.array([M.ZONE_INDEX[c] for c in M.parkes_zone(np.full(300, r), p, t)])
                assert idx[0] == 0 and (np.diff(idx) >= 0).all()
    # the diagonal is zone A on both grids, and the two grids genuinely differ
    r = rng.uniform(1, 549, 2000)
    assert (M.parkes_zone(r, r, 1) == "A").all() and (M.parkes_zone(r, r, 2) == "A").all()
    p = rng.uniform(0, 600, 2000)
    assert (M.parkes_zone(r, p, 1) != M.parkes_zone(r, p, 2)).any()
    with pytest.raises(ValueError):
        M.parkes_zone([1], [1], diabetes_type=3)


def test_parkes_summary_keys():
    s = M.parkes_summary(np.linspace(40, 400, 50), np.linspace(40, 400, 50))
    assert s["parkes_zone_A_pct"] == 100.0 and s["total_points"] == 50
    assert set(s) == {"total_points", *[f"parkes_zone_{z}_{k}" for z in M.ZONES for k in ("count", "pct")]}


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def test_event_metrics():
    r = np.array([60, 65, 100, 200, 250, 120])
    p = np.array([68, 90, 60, 150, 260, 130])
    e = M.event_metrics(r, p)
    assert e["hypo"] == {"n_events": 2, "tp": 1, "fp": 1, "fn": 1, "tn": 3, "sensitivity": 0.5,
                         "specificity": 0.75, "ppv": 0.5, "npv": 0.75}
    assert e["hyper"]["n_events"] == 2 and e["hyper"]["tp"] == 1 and e["hyper"]["fp"] == 0
    none = M.event_metrics([100, 110], [100, 110])
    assert none["hypo"]["sensitivity"] is None and none["hypo"]["specificity"] == 1.0


# --------------------------------------------------------------------------- #
# record_level / summarize
# --------------------------------------------------------------------------- #
def test_record_level_averages_windows_and_drops_nan():
    ws = _WS(record=["r1", "r1", "r2", "r2", "r3"], subject=["s", "s", "s", "t", "t"], y=[100, 100, 120, 120, 90])
    pred = np.array([90.0, 110.0, np.nan, 130.0, 95.0])
    df = M.record_level(pred, ws)
    assert df.columns.tolist() == ["record", "subject", "y", "pred", "n_windows"]
    assert df.set_index("record").pred.to_dict() == {"r1": 100.0, "r2": 130.0, "r3": 95.0}
    assert df.set_index("record").n_windows.to_dict() == {"r1": 2, "r2": 1, "r3": 1}
    assert len(M.record_level(pred, ws, drop_nan=False)) == 4          # (r2, s) and (r2, t) rows
    with pytest.raises(ValueError):
        M.record_level(pred[:3], ws)


def test_record_level_matches_legacy():
    legacy = pytest.importorskip("run_subject_cv")
    rng = np.random.default_rng(0)
    n = 60
    ws = _WS(record=[f"r{i // 3}" for i in range(n)], subject=[f"s{i // 9}" for i in range(n)],
             y=np.repeat(rng.uniform(80, 200, n // 3), 3))
    pred = rng.uniform(80, 200, n).astype(np.float32)
    pd.testing.assert_frame_equal(M.record_level(pred, ws), legacy.recording_level(pred, ws))


def test_summarize_keys_and_values(table):
    m = M.summarize(table)
    first = ["n", "mae", "rmse", "mard_pct", "r2", "pearson_r", "within_subject_r_median"]
    assert list(m)[:7] == first
    assert m["n"] == 20 and m["n_subjects"] == 4
    assert m["mae"] == pytest.approx(np.abs(table.pred - table.y).mean())
    assert sum(m[f"zone_{z}_pct"] for z in M.ZONES) == pytest.approx(100)
    assert sum(m[f"parkes_zone_{z}_pct"] for z in M.ZONES) == pytest.approx(100)
    assert m["zone_A_pct"] == 100.0 and m["parkes_zone_A_pct"] == 100.0
    assert "hypo" in m and "hyper" in m and "hypo" not in M.summarize(table, events=False)
    assert m["bias"] == pytest.approx((table.pred - table.y).mean())


def test_summarize_degenerate():
    one = pd.DataFrame({"record": ["a"], "subject": ["s"], "y": [100.0], "pred": [90.0]})
    assert M.summarize(one) == {"n": 1, "mae": 10.0, "error": "degenerate"}
    flat = pd.DataFrame({"record": list("ab"), "subject": list("ss"), "y": [100.0, 100.0], "pred": [90.0, 120.0]})
    assert M.summarize(flat)["error"] == "degenerate"
    empty = pd.DataFrame({"record": [], "subject": [], "y": [], "pred": []})
    assert M.summarize(empty) == {"n": 0, "mae": None, "error": "degenerate"}


def test_summarize_matches_legacy_bitwise():
    legacy = pytest.importorskip("run_subject_cv")
    files = sorted(glob.glob(os.path.join(RESULTS, "subject_cv_*_seed0.csv")))[:4]
    frames = [pd.read_csv(f) for f in files]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"record": [f"{i // 3:03d}" for i in range(90)], "subject": [f"{i // 9:02d}" for i in range(90)],
                       "y": rng.uniform(70, 250, 90).astype(np.float32),
                       "pred": rng.uniform(60, 260, 90).astype(np.float32)})
    frames.append(df)                                   # float32 columns, as inside a run
    frames.append(df.assign(pred=df.y.mean()))          # constant predictor -> nan correlation
    for f in frames:
        a, b = M.summarize(f), legacy.metrics(f)
        for k, v in b.items():
            if isinstance(v, float) and np.isnan(v):
                assert np.isnan(a[k]), k
            else:
                assert a[k] == v, k


def test_per_subject(table):
    ps = M.per_subject(table, min_records=5)
    assert set(ps) == {"00", "01", "02", "03"} and all("mae" in v for v in ps.values())
    assert M.per_subject(table, min_records=6) == {}


def test_aggregate_and_pool_seeds(table):
    runs = [{"mae": 10.0, "r2": 0.5, "n": 3}, {"mae": 12.0, "r2": None, "n": 3}, {"error": "degenerate"}]
    agg = M.aggregate_seeds(runs)
    assert agg["mae"] == {"mean": 11.0, "std": 1.0, "n": 2, "values": [10.0, 12.0]}
    assert agg["r2"]["n"] == 1 and agg["n"]["n"] == 2
    assert M.aggregate_seeds([{"error": "degenerate"}]) == {}
    t2 = table.assign(pred=table.pred + 2, subject=table.subject.astype(int))   # CSV round-trip loses zero padding
    pooled = M.pool_seeds([table, t2])
    assert len(pooled) == len(table) and (pooled.subject.str.len() == 3).all()
    merged = pooled.merge(table, on="record", suffixes=("", "_a"))
    assert np.allclose(merged.pred, merged.pred_a + 1)


# --------------------------------------------------------------------------- #
# Cluster bootstrap
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_basic(table):
    null = table.assign(pred=table.y.mean())
    out = M.cluster_bootstrap(table, null, B=300, seed=0)
    assert set(out) == {"mae", "mae_ci", "diff_vs_null", "diff_ci", "p_better_than_null", "n_subjects", "n_records"}
    assert out["n_subjects"] == 4 and out["n_records"] == 20
    assert out["mae_ci"][0] <= out["mae"] <= out["mae_ci"][1]
    assert out["diff_vs_null"] < 0 and out["p_better_than_null"] > 0.9
    assert out == M.cluster_bootstrap(table, null, B=300, seed=0)            # deterministic
    big = pd.concat([table.assign(subject=table.subject + str(k), record=table.record + str(k),
                                  pred=table.pred + k) for k in range(5)])   # 20 subjects: seeds differ
    assert M.cluster_bootstrap(big, big.assign(pred=big.y.mean()), B=300, seed=0) != \
        M.cluster_bootstrap(big, big.assign(pred=big.y.mean()), B=300, seed=1)
    same = M.cluster_bootstrap(table, table, B=50)
    assert same["diff_vs_null"] == 0 and same["diff_ci"] == [0.0, 0.0] and same["p_better_than_null"] == 0.0
    one = table[table.subject == "00"]
    assert M.cluster_bootstrap(one, one.assign(pred=100.0), B=10)["error"] == "degenerate"
    with pytest.raises(ValueError):
        M.cluster_bootstrap(table, null.assign(record="x" + null.record), B=10)


def _load_legacy(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    return M.pool_seeds([pd.read_csv(f) for f in files]) if files else None


def test_cluster_bootstrap_matches_bootstrap_ci_json():
    """Same draws as bootstrap_ci.py: first two models with one generator threaded through."""
    path = os.path.join(RESULTS, "bootstrap_ci.json")
    null, ridge, demo = (_load_legacy(p) for p in
                         ("subject_cv_mean_seed0.csv", "subject_cv_ridge_seed*.csv", "subject_cv_demo_seed*.csv"))
    if not os.path.exists(path) or null is None or ridge is None or demo is None:
        pytest.skip("legacy result files not present")
    ref = json.load(open(path))
    if "mendeley/ridge" not in ref or "mendeley/demo" not in ref:
        pytest.skip("legacy bootstrap_ci.json lacks the mendeley entries")
    rng = np.random.default_rng(0)
    assert M.cluster_bootstrap(ridge, null, rng=rng) == ref["mendeley/ridge"]
    assert M.cluster_bootstrap(demo, null, rng=rng) == ref["mendeley/demo"]
    assert M.cluster_bootstrap(ridge, null, seed=0) == ref["mendeley/ridge"]   # fresh generator == first entry


def test_cluster_bootstrap_metrics_and_paired(table):
    ci = M.cluster_bootstrap_metrics(table, B=100, seed=0)
    assert ci["n_subjects"] == 4 and ci["B"] == 100
    for k in ("mae", "pearson_r", "zone_A_pct", "parkes_zone_A_pct"):
        assert ci[k]["lo"] <= ci[k]["point"] <= ci[k]["hi"] and ci[k]["n_boot"] > 0
    assert M.cluster_bootstrap_metrics(table[table.subject == "00"], B=5)["error"] == "degenerate"
    worse = table.assign(pred=table.pred + 20)
    pr = M.paired_cluster_bootstrap(table, worse, "mae", B=200, seed=0)
    assert pr["diff"] < 0 and pr["diff_ci"][1] < 0 and pr["p_a_better"] == 1.0
    pr_r = M.paired_cluster_bootstrap(table, worse, metric=lambda y, p: float(np.abs(p - y).max()), B=20)
    assert pr_r["metric"] == "<lambda>" and pr_r["diff"] < 0
    with pytest.raises(ValueError):
        M.paired_cluster_bootstrap(table, worse.assign(record="x"), B=5)
