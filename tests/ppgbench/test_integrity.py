"""
Tests for :mod:`ppgbench.integrity`.

The label-cycling tests construct columns with exactly the structure of the
corrupted ``Archive_Dataset/clean-dataset.csv`` (per subject, the glucose
label cycles row by row through that subject's reference values, with a
different period per subject) and check that genuine run-structured labels
are *not* flagged.  The audit tests use the ``synthetic`` dataset and the
``protocols`` module when they are importable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppgbench import integrity as I  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cycling_column(values: list[float], n_rows: int) -> np.ndarray:
    """Label column with the clean-dataset.csv artefact: ``values`` repeated row by row."""
    return np.resize(np.asarray(values, dtype=float), n_rows)


def run_column(values: list[float], windows_per_record: int) -> np.ndarray:
    """Genuine structure: each record's label repeated for its windows, records in order."""
    return np.repeat(np.asarray(values, dtype=float), windows_per_record)


class _WS:
    """Duck-typed WindowSet for tests that do not need the real class."""

    def __init__(self, X, y, subject, record, time=None, meta=None, source="fake"):
        self.X, self.y = np.asarray(X, np.float32), np.asarray(y, np.float32)
        self.subject, self.record = np.asarray(subject, str), np.asarray(record, str)
        self.time, self.meta, self.source = time, (pd.DataFrame() if meta is None else meta), source

    def __len__(self):
        return len(self.y)


# --------------------------------------------------------------------------- #
# detect_label_cycling
# --------------------------------------------------------------------------- #
def test_detects_cycling_single_subject():
    lab = cycling_column([99, 102, 103, 128, 130, 134, 136], 1000)       # subject 1 of the corrupted CSV
    res = I.detect_label_cycling(lab)
    assert res["cycling"] and res["period"] == 7 and res["kind"] == "cycling"
    assert res["fraction_matching"] > 0.99 and res["n_unique"] == 7
    assert res["fractions"][1] == 0.0 and res["fractions"][7] == res["fraction_matching"]


def test_detects_cycling_per_subject_with_different_periods():
    """The real artefact: period 7 for subject 1, 3 for subject 2, 2 for subject 3 ... so the
    global test fails but every subject fires."""
    subjects = {"1": [99, 102, 103, 128, 130, 134, 136], "2": [108, 111, 118], "3": [120, 127],
                "5": [94, 96, 106, 110, 129, 136]}
    labels = np.concatenate([cycling_column(v, 400 + 50 * k) for k, v in enumerate(subjects.values())])
    groups = np.concatenate([np.full(400 + 50 * k, s) for k, s in enumerate(subjects)])
    res = I.detect_label_cycling(labels, groups=groups)
    assert res["cycling"] and res["period"] is None                      # no global period, as in the CSV
    assert res["n_groups"] == 4 and res["n_groups_cycling"] == 4
    assert {k: v["period"] for k, v in res["per_group"].items()} == {"1": 7, "2": 3, "3": 2, "5": 6}
    assert all(v["kind"] == "cycling" for v in res["per_group"].values())
    # add a legitimate subject: it must not be flagged
    legit = run_column([101, 97, 133, 88, 120, 110, 95, 140], 3)
    res = I.detect_label_cycling(np.concatenate([labels, legit]), groups=np.concatenate([groups, np.full(len(legit), "9")]))
    assert res["groups_cycling"] == ["1", "2", "3", "5"] and res["per_group"]["9"]["kind"] is None


def test_detects_cycling_with_string_and_int_labels():
    lab = np.array(["a", "b", "c"] * 100)
    assert I.detect_label_cycling(lab)["period"] == 3
    assert I.detect_label_cycling(np.tile([1, 2], 50))["period"] == 2


def test_genuine_labels_are_not_cycling():
    rng = np.random.default_rng(0)
    # Mendeley-like: 67 records x 3 overlapping windows, one label per record
    labels = run_column(rng.uniform(80, 200, 67), 3)
    res = I.detect_label_cycling(labels)
    assert not res["cycling"] and res["period"] is None
    assert res["fractions"][1] == pytest.approx((67 * 2) / (67 * 3 - 1))     # runs, not cycles
    # many windows per record (12): still well below the threshold
    assert not I.detect_label_cycling(run_column(rng.uniform(80, 200, 100), 12))["cycling"]
    # random labels
    assert not I.detect_label_cycling(rng.integers(70, 250, 2000))["cycling"]


def test_constant_column_is_constant_not_cycling():
    res = I.detect_label_cycling(np.full(100, 110.0))
    assert res["period"] == 1 and res["kind"] == "constant" and not res["cycling"]
    res = I.detect_label_cycling(np.full(100, 110.0), groups=np.repeat(["a", "b"], 50))
    assert not res["cycling"] and res["n_groups_cycling"] == 0


def test_cycling_edge_cases():
    assert I.detect_label_cycling(np.array([1.0, 2.0, 3.0]))["kind"] == "too_short"
    short = I.detect_label_cycling(np.tile([1.0, 2.0], 3), min_rows=2)
    assert short["period"] == 2
    assert I.detect_label_cycling(np.tile([1.0, 2.0, 3.0], 20), max_period=2)["period"] is None
    with pytest.raises(ValueError):
        I.detect_label_cycling(np.arange(10), groups=np.arange(9))
    nan = np.tile([1.0, np.nan], 20)
    assert I.detect_label_cycling(nan)["period"] is None                   # NaN never matches


# --------------------------------------------------------------------------- #
# detect_duplicate_windows
# --------------------------------------------------------------------------- #
def test_duplicate_windows_exact_and_shifted():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    L, hop = 320, 160
    starts = [0, 160, 320, 1000, 1500]
    X = np.stack([x[s:s + L] for s in starts])
    X = np.vstack([X, X[3] * 3.0 + 5.0, X[4]])                           # scaled copy, exact copy
    pairs = I.detect_duplicate_windows(X, hop=hop)
    assert (3, 5) in pairs and (4, 6) in pairs                            # duplicates (scale-invariant)
    assert (0, 1) in pairs and (1, 2) in pairs and (0, 2) not in pairs     # hop-shifted neighbours only
    assert (3, 4) not in pairs
    exact_only = I.detect_duplicate_windows(X)
    assert exact_only == [(3, 5), (4, 6)]
    assert I.detect_duplicate_windows(X, normalize=False) == [(4, 6)]
    assert I.detect_duplicate_windows(X[:1]) == []
    with pytest.raises(ValueError):
        I.detect_duplicate_windows(X, hop=L)
    with pytest.raises(ValueError):
        I.detect_duplicate_windows(X.ravel())


def test_duplicate_windows_tolerance():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 64))
    Y = np.vstack([X, X[0] + 1e-12])
    assert (0, 5) in I.detect_duplicate_windows(Y, tol=1e-6)
    noisy = np.vstack([X, X[0] + 1e-3 * rng.standard_normal(64)])
    assert I.detect_duplicate_windows(noisy, tol=1e-6) == []
    assert I.detect_duplicate_windows(noisy, tol=1e-6, normalize=False) == []
    # an offset / rescaled copy is the same signal after per-window normalisation
    assert I.detect_duplicate_windows(np.vstack([X, X[0] + 0.1]), tol=1e-6) == [(0, 5)]
    assert I.detect_duplicate_windows(np.vstack([X, X[0] + 0.1]), tol=1e-6, normalize=False) == []


# --------------------------------------------------------------------------- #
# check_timestamps
# --------------------------------------------------------------------------- #
def test_timestamps_regular_cgm():
    t = 1.6e9 + 300.0 * np.arange(200)
    res = I.check_timestamps(t)
    assert res["ok"] and res["monotone"] and res["fraction_regular"] == 1.0 and res["n_gaps"] == 0
    assert res["median_spacing_s"] == 300.0 and res["unit_suspect"] is None and res["problems"] == []


def test_timestamps_detect_unit_errors_gaps_and_order():
    t = 1.6e9 + 300.0 * np.arange(200)
    ms = I.check_timestamps(t * 1000)
    assert not ms["ok"] and "milliseconds" in ms["unit_suspect"]
    small = I.check_timestamps(t / 1000)
    assert not small["ok"] and "too small" in small["unit_suspect"]
    gappy = I.check_timestamps(np.concatenate([t[:100], t[100:] + 7200]))
    assert gappy["ok"] and gappy["n_gaps"] == 1 and gappy["max_gap_s"] == 7500.0
    back = I.check_timestamps(np.concatenate([t[:100], t[50:60], t[100:]]))
    assert not back["ok"] and back["n_backwards"] == 1 and back["n_duplicates"] == 10
    irregular = I.check_timestamps(np.sort(np.random.default_rng(0).uniform(0, 1e5, 200)))
    assert not irregular["ok"] and irregular["fraction_regular"] < 0.5
    assert I.check_timestamps(np.array([5.0]))["ok"]


def test_timestamps_per_group():
    t = 1.6e9 + 300.0 * np.arange(100)
    groups = np.repeat(["a", "b"], 50)
    tt = np.concatenate([t[:50], t[:50]])                                   # two subjects, same clock -> fine per group
    res = I.check_timestamps(tt, groups=groups)
    assert res["ok"] and set(res["per_group"]) == {"a", "b"} and res["per_group"]["b"]["fraction_regular"] == 1.0
    bad = I.check_timestamps(np.concatenate([t[:50], t[:50] * 1000]), groups=groups)
    assert not bad["ok"] and any("group b" in p for p in bad["problems"])
    with pytest.raises(ValueError):
        I.check_timestamps(t, groups=groups[:10])


# --------------------------------------------------------------------------- #
# check_record_labels
# --------------------------------------------------------------------------- #
def test_check_record_labels():
    X = np.zeros((6, 8))
    ok = _WS(X, [1, 1, 2, 2, 3, 3], list("aabbcc"), ["r1", "r1", "r2", "r2", "r3", "r3"],
             meta=pd.DataFrame({"record": ["r1", "r2", "r3"], "subject": list("abc"), "glucose": [1.0, 2.0, 3.0]}))
    assert I.check_record_labels(ok)["ok"]
    bad = _WS(X, [1, 5, 2, 2, 3, 3], list("aabbcd"), ["r1", "r1", "r2", "r2", "r3", "r3"],
              meta=pd.DataFrame({"record": ["r1", "r2", "r3"], "subject": list("abc"), "glucose": [1.0, 2.0, 9.0]}))
    res = I.check_record_labels(bad)
    assert not res["ok"] and res["multi_label_records"] == ["r1"] and res["multi_subject_records"] == ["r3"]
    assert res["meta_mismatch"] == ["r3"]


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _fake_ws(n_subjects=3, records=6, wpr=3, seed=0, with_time=True):
    rng = np.random.default_rng(seed)
    rows, X, y, sub, rec, t, meta = 0, [], [], [], [], [], []
    for s in range(n_subjects):
        for r in range(records):
            g = float(rng.uniform(80, 200))
            key = f"{s:03d}_{r:05d}"
            tt = 1.6e9 + 300.0 * r
            for _ in range(wpr):
                X.append(rng.standard_normal(32)); y.append(g); sub.append(f"{s:03d}"); rec.append(key); t.append(tt)
            meta.append({"record": key, "subject": f"{s:03d}", "glucose": g, "egv_time": tt})
    meta = pd.DataFrame(meta)
    if not with_time:
        meta = meta.drop(columns="egv_time")
    return _WS(np.array(X), y, sub, rec, time=np.array(t) if with_time else None, meta=meta)


def test_audit_clean_fake_set_without_folds():
    rep = I.audit(_fake_ws())
    assert rep["ok"] and rep["problems"] == [] and rep["leak"] is None
    assert rep["labels"]["windows"]["cycling"] is False and rep["labels"]["records"]["cycling"] is False
    assert rep["records"]["ok"] and rep["duplicates"]["n_cross_record"] == 0
    assert rep["timestamps"]["ok"] and rep["n_records"] == 18 and rep["n_subjects"] == 3
    assert I.audit(_fake_ws(with_time=False))["timestamps"] is None
    no_t = _fake_ws(with_time=False)
    no_t.meta = _fake_ws().meta                                             # times only in meta.egv_time -> used
    assert I.audit(no_t)["timestamps"]["ok"]
    assert I.audit_ok(_fake_ws())


def test_audit_flags_cross_join_artefact():
    ws = _fake_ws(wpr=1)
    # cross-join: every window duplicated under every label of its subject (labels then cycle)
    idx = np.concatenate([np.repeat(np.flatnonzero(ws.subject == s), 6) for s in np.unique(ws.subject)])
    lab = np.concatenate([np.resize(ws.y[ws.subject == s], (ws.subject == s).sum() * 6) for s in np.unique(ws.subject)])
    bad = _WS(ws.X[idx], lab, ws.subject[idx], ws.record[idx], meta=ws.meta)
    rep = I.audit(bad)
    assert not rep["ok"]
    assert rep["labels"]["windows"]["cycling"] and rep["labels"]["windows"]["n_groups_cycling"] == 3
    assert rep["duplicates"]["n_conflicting_label"] > 0
    assert not rep["records"]["ok"]                                         # one record now carries 6 labels
    assert any("label cycling" in p for p in rep["problems"])


def test_audit_with_protocols_leak_check():
    P = pytest.importorskip("ppgbench.protocols")
    ws = _fake_ws()
    subjects = np.unique(ws.subject)
    folds = [(np.flatnonzero(ws.subject != s), np.flatnonzero(ws.subject == s)) for s in subjects]
    rep = I.audit(ws, folds, "subject_loso", hop=None)
    assert rep["ok"] and rep["leak"]["error"] is None and rep["leak"]["passes"]["subject"] is True
    assert rep["leak"]["n_folds"] == 3 and rep["leak"]["test_coverage"] == 1.0
    leaky = [(np.arange(len(ws))[::2], np.arange(len(ws))[1::2])]           # windows of one record on both sides
    rep = I.audit(ws, leaky, "subject_loso")
    assert not rep["ok"] and rep["leak"]["error"] and rep["leak"]["passes"]["record"] is False
    assert any(p.startswith("leak at level subject") for p in rep["problems"])
    with pytest.raises(P.LeakError):
        I.assert_no_leak(ws, leaky, "record")
    I.assert_no_leak(ws, folds, "subject")
    assert I.leak_report(ws, folds, "subject_loso")["passes"]["subject"] is True
    assert I.LeakError is P.LeakError


def test_audit_on_synthetic_dataset_with_make_folds():
    registry = pytest.importorskip("ppgbench.data.registry")
    P = pytest.importorskip("ppgbench.protocols")
    ws = registry.load_dataset("synthetic", n_subjects=6, records_per_subject=6, windows_per_record=3, seed=0)
    for protocol in ("subject_loso", "subject_kfold", "record_random", "chronological_nogap"):
        folds = P.make_folds(ws, protocol)
        rep = I.audit(ws, folds, protocol)
        assert rep["ok"], (protocol, rep["problems"])
    assert I.audit(ws, P.make_folds(ws, "window_random"), "window_random")["ok"]


# --- block-wise (partial) cross-join detector --------------------------------
def test_local_cycling_detects_partial_corruption_that_global_misses():
    import numpy as np
    from ppgbench.integrity import detect_label_cycling, detect_label_cycling_local
    rng = np.random.default_rng(1)
    k, run = 10, 200
    vals = np.round(rng.normal(115, 18, size=k))
    clean = np.repeat(vals, run)
    col = clean.copy()
    n_block = len(col) // 4                                   # only 25 % of the rows cycled
    col[:n_block] = vals[np.arange(n_block) % k]
    assert not detect_label_cycling(col)["cycling"]           # dilution hides it from the whole-column test
    assert detect_label_cycling_local(col)["cycling"]         # the block scan finds it


def test_local_cycling_has_no_false_alarm_on_record_structured_labels():
    import numpy as np
    from ppgbench.integrity import detect_label_cycling_local
    rng = np.random.default_rng(2)
    for run in (3, 50, 400):                                  # short and long runs of one label per record
        col = np.repeat(np.round(rng.normal(115, 18, size=40)), run)
        assert not detect_label_cycling_local(col, block=256)["cycling"]


def test_local_cycling_rejects_too_small_block():
    import pytest
    from ppgbench.integrity import detect_label_cycling_local
    with pytest.raises(ValueError):
        detect_label_cycling_local([1, 2, 3], block=16)
