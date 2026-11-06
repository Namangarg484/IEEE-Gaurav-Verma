"""
Tests for ``ppgbench.protocols``.

Run:  ~/.venvs/ppgfusion/bin/python -m pytest tests/ppgbench/test_protocols.py -q

Everything runs on a tiny deterministic synthetic WindowSet (no real data,
no training).  For every protocol there is (a) an index-for-index parity
check against a verbatim copy of the legacy split code, (b) a clean-split
test that must pass ``assert_no_leak`` at the declared level and (c) a
deliberately leaky split that must raise ``LeakError``.  The parity test
against the *imported* legacy modules is marked ``slow`` (set
``PPGBENCH_SLOW=1``) because importing them pulls in torch.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppgbench import protocols as P  # noqa: E402
from ppgbench.protocols import (  # noqa: E402
    LEGACY_ALIASES, LEVELS, PROTOCOLS, LeakError, assert_no_leak, chronological_split, declared_level,
    fold_fingerprint, folds_from_json, folds_to_json, leak_report, make_folds, verify_folds, window_time,
)

SLOW = pytest.mark.skipif(os.environ.get("PPGBENCH_SLOW") != "1", reason="set PPGBENCH_SLOW=1")


# --------------------------------------------------------------------------- #
# Synthetic WindowSet (real class when ppgbench.core exists, duck-typed otherwise)
# --------------------------------------------------------------------------- #
@dataclass
class _DuckWindowSet:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    record: np.ndarray
    time: np.ndarray | None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    fs: int = 64
    source: str = ""

    def __len__(self) -> int:
        return len(self.y)


def _windowset(**fields):
    try:
        from ppgbench.core.windowset import WindowSet
        return WindowSet(**fields)
    except (ImportError, TypeError):
        return _DuckWindowSet(**fields)


def make_ws(n_subjects: int = 5, records_per_subject: int = 40, windows_per_record: int = 5, L: int = 32,
            spacing_s: float = 300.0, seed: int = 0, with_time: bool = True, small_subject: bool = True):
    """Subjects S00.., records S00_R000.., ``windows_per_record`` windows per record.

    Reading times are ``spacing_s`` apart and monotone within a subject (each
    subject starts a day later).  With ``small_subject`` the last subject has
    only four records so that ``personal`` must skip it.
    """
    rng = np.random.default_rng(seed)
    X, y, subject, record, time, meta = [], [], [], [], [], []
    for s in range(n_subjects):
        n_rec = 4 if (small_subject and s == n_subjects - 1) else records_per_subject
        base = 1.6e9 + s * 86400.0
        for r in range(n_rec):
            key = f"S{s:02d}_R{r:03d}"
            g = 100.0 + 15.0 * s + float(rng.normal(0, 10))
            t = base + r * spacing_s
            for _ in range(windows_per_record):
                X.append(rng.normal(size=L).astype(np.float32))
                y.append(g); subject.append(f"S{s:02d}"); record.append(key); time.append(t)
            meta.append({"record": key, "subject": f"S{s:02d}", "glucose": g, "egv_time": t,
                         "n_windows": windows_per_record})
    return _windowset(X=np.stack(X), y=np.asarray(y, np.float32), subject=np.asarray(subject),
                      record=np.asarray(record), time=np.asarray(time, np.float64) if with_time else None,
                      meta=pd.DataFrame(meta), fs=64, source="synthetic-test")


@pytest.fixture(scope="module")
def ws():
    return make_ws()


@pytest.fixture(scope="module")
def ws_notime():
    return make_ws(with_time=False)


# --------------------------------------------------------------------------- #
# Verbatim copies of the legacy split code (reference behaviour)
# --------------------------------------------------------------------------- #
def legacy_mendeley_folds(ws, protocol, seed):                       # run_leakage_ladder.mendeley_folds
    if protocol == "window":
        return list(KFold(5, shuffle=True, random_state=seed).split(ws.X))
    if protocol == "record":
        rng = np.random.RandomState(seed)
        recs = np.unique(ws.record); rng.shuffle(recs)
        grp = {r: i % 5 for i, r in enumerate(recs)}
        g = np.array([grp[r] for r in ws.record])
        return [(np.flatnonzero(g != k), np.flatnonzero(g == k)) for k in range(5)]
    if protocol == "subject":
        return list(LeaveOneGroupOut().split(ws.X, ws.y, ws.subject))
    raise ValueError(protocol)


def legacy_chronological_split(ws, train_frac: float = 0.7, gap_s: float = 3600.0):   # run_bigideas_cv
    rec_time = ws.meta.set_index("record").egv_time
    t = rec_time.loc[ws.record].to_numpy()
    tr, te = np.zeros(len(ws), bool), np.zeros(len(ws), bool)
    for s in np.unique(ws.subject):
        m = ws.subject == s
        cut = np.quantile(np.unique(t[m]), train_frac)
        tr |= m & (t < cut - gap_s)
        te |= m & (t >= cut)
    return np.flatnonzero(tr), np.flatnonzero(te)


def legacy_bigideas_folds(ws, protocol, seed):                       # run_leakage_ladder.bigideas_folds
    if protocol == "reading":
        return list(GroupKFold(5).split(ws.X, ws.y, ws.record))
    if protocol == "chrono0":
        return [legacy_chronological_split(ws, gap_s=0.0)]
    if protocol == "chrono":
        return [legacy_chronological_split(ws, gap_s=3600.0)]
    raise ValueError(protocol)


def legacy_personal(ws):                                             # run_bigideas_cv.main, --protocol personal
    tr_all, te_all = legacy_chronological_split(ws)
    folds = [(tr_all[ws.subject[tr_all] == s], te_all[ws.subject[te_all] == s]) for s in np.unique(ws.subject)]
    return [(tr, te) for tr, te in folds if len(tr) > 64 and len(te)]


def legacy_subject_cv_folds(ws, folds):                              # run_subject_cv.main
    splitter = LeaveOneGroupOut() if folds == "loso" else GroupKFold(int(folds))
    return list(splitter.split(ws.X, ws.y, ws.subject))


def assert_same_folds(a, b):
    assert len(a) == len(b), f"{len(a)} folds vs {len(b)}"
    for k, ((tr1, te1), (tr2, te2)) in enumerate(zip(a, b)):
        np.testing.assert_array_equal(tr1, tr2, err_msg=f"fold {k} train differs")
        np.testing.assert_array_equal(te1, te2, err_msg=f"fold {k} test differs")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
SPEC_PROTOCOLS = {"window_random": "window", "record_random": "record", "subject_kfold": "subject",
                  "subject_loso": "subject", "chronological": "time", "chronological_nogap": "time",
                  "personal": "time"}


def test_registry_has_every_spec_protocol_with_its_level():
    for name, level in SPEC_PROTOCOLS.items():
        assert name in PROTOCOLS
        assert declared_level(name) == level
    assert set(P.list_protocols()) == set(PROTOCOLS)
    assert all(lvl in LEVELS for lvl in (s.level for s in PROTOCOLS.values()))


def test_legacy_aliases_resolve():
    for alias, name in LEGACY_ALIASES.items():
        assert P.resolve_protocol(alias).name == name
    with pytest.raises(ValueError, match="unknown protocol"):
        make_folds(make_ws(n_subjects=2, records_per_subject=3, windows_per_record=1), "no_such_protocol")


def test_unknown_parameter_is_a_type_error(ws):
    with pytest.raises(TypeError, match="unexpected keyword"):
        make_folds(ws, "subject_loso", bogus=1)


# --------------------------------------------------------------------------- #
# Parity with the legacy scripts (verbatim copies above)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_window_random_matches_legacy(ws, seed):
    assert_same_folds(make_folds(ws, "window_random", seed=seed), legacy_mendeley_folds(ws, "window", seed))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_record_random_matches_legacy(ws, seed):
    assert_same_folds(make_folds(ws, "record_random", seed=seed), legacy_mendeley_folds(ws, "record", seed))


def test_subject_loso_matches_legacy(ws):
    assert_same_folds(make_folds(ws, "subject_loso"), legacy_mendeley_folds(ws, "subject", 0))
    assert_same_folds(make_folds(ws, "loso"), legacy_subject_cv_folds(ws, "loso"))


@pytest.mark.parametrize("k", [3, 5])
def test_subject_kfold_matches_legacy(ws, k):
    assert_same_folds(make_folds(ws, "subject_kfold", n_splits=k), legacy_subject_cv_folds(ws, str(k)))


def test_record_kfold_matches_bigideas_reading(ws):
    assert_same_folds(make_folds(ws, "record_kfold"), legacy_bigideas_folds(ws, "reading", 0))
    assert_same_folds(make_folds(ws, "reading"), legacy_bigideas_folds(ws, "reading", 0))


@pytest.mark.parametrize("gap_s", [0.0, 3600.0, 900.0])
@pytest.mark.parametrize("train_frac", [0.7, 0.5])
def test_chronological_split_matches_legacy(ws, gap_s, train_frac):
    tr, te = chronological_split(ws, train_frac=train_frac, gap_s=gap_s)
    tr2, te2 = legacy_chronological_split(ws, train_frac=train_frac, gap_s=gap_s)
    np.testing.assert_array_equal(tr, tr2)
    np.testing.assert_array_equal(te, te2)
    assert tr.dtype == np.int64 and te.dtype == np.int64


def test_chronological_protocols_match_legacy(ws):
    assert_same_folds(make_folds(ws, "chronological"), legacy_bigideas_folds(ws, "chrono", 0))
    assert_same_folds(make_folds(ws, "chrono"), legacy_bigideas_folds(ws, "chrono", 0))
    assert_same_folds(make_folds(ws, "within"), legacy_bigideas_folds(ws, "chrono", 0))
    assert_same_folds(make_folds(ws, "chronological_nogap"), legacy_bigideas_folds(ws, "chrono0", 0))
    assert_same_folds(make_folds(ws, "chrono0"), legacy_bigideas_folds(ws, "chrono0", 0))


def test_chronological_split_uses_quantile_of_unique_times_with_guard_gap(ws):
    """Explicit check of the rule, independent of the legacy copy."""
    t = window_time(ws)
    tr, te = chronological_split(ws, train_frac=0.7, gap_s=3600.0)
    for s in np.unique(ws.subject):
        m = ws.subject == s
        cut = np.quantile(np.unique(t[m]), 0.7)
        assert (t[tr][ws.subject[tr] == s] < cut - 3600.0).all()
        assert (t[te][ws.subject[te] == s] >= cut).all()
        # every window of this subject is train, test or in the gap -- never anywhere else
        gap = m & (t >= cut - 3600.0) & (t < cut)
        assert m.sum() == (ws.subject[tr] == s).sum() + (ws.subject[te] == s).sum() + gap.sum()


def test_chronological_falls_back_to_meta_egv_time(ws_notime):
    """Legacy WindowSets carry the time only in meta['egv_time']."""
    assert window_time(ws_notime) is not None
    assert_same_folds(make_folds(ws_notime, "chronological"), legacy_bigideas_folds(ws_notime, "chrono", 0))


def test_personal_matches_legacy_and_skips_small_subjects(ws):
    folds = make_folds(ws, "personal")
    assert_same_folds(folds, legacy_personal(ws))
    held = [np.unique(ws.subject[te]) for _, te in folds]
    assert all(len(h) == 1 for h in held)
    assert all(set(np.unique(ws.subject[tr])) == set(h) for (tr, _), h in zip(folds, held))
    assert "S04" not in {h[0] for h in held}, "the 4-record subject must be skipped (<= 64 train windows)"
    assert len(folds) == 4
    assert all(len(tr) > 64 for tr, _ in folds)
    assert len(make_folds(ws, "personal", min_train=1)) == 4          # S04 spans 900 s: no train before the 1-h gap
    assert len(make_folds(ws, "personal", min_train=1, gap_s=0.0)) == 5


def test_personal_raises_when_nobody_qualifies(ws):
    with pytest.raises(ValueError, match="no subject"):
        make_folds(ws, "personal", min_train=10_000)


@SLOW
def test_parity_against_imported_legacy_modules(ws):
    """Imports run_leakage_ladder / run_bigideas_cv / run_subject_cv (needs torch)."""
    import run_bigideas_cv
    import run_leakage_ladder
    from sklearn.model_selection import GroupKFold as GKF, LeaveOneGroupOut as LOGO  # noqa: F401
    for proto, alias in (("window_random", "window"), ("record_random", "record"), ("subject_loso", "subject")):
        for seed in (0, 1):
            assert_same_folds(make_folds(ws, proto, seed=seed), run_leakage_ladder.mendeley_folds(ws, alias, seed))
    for proto, alias in (("record_kfold", "reading"), ("chronological_nogap", "chrono0"), ("chronological", "chrono")):
        assert_same_folds(make_folds(ws, proto), run_leakage_ladder.bigideas_folds(ws, alias, 0))
    tr, te = run_bigideas_cv.chronological_split(ws)
    np.testing.assert_array_equal(make_folds(ws, "chronological")[0][0], tr)
    np.testing.assert_array_equal(make_folds(ws, "chronological")[0][1], te)


# --------------------------------------------------------------------------- #
# Every protocol: clean folds pass at the declared level
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PROTOCOLS))
def test_protocol_folds_are_well_formed_and_pass_declared_level(ws, name):
    folds = make_folds(ws, name)                       # make_folds already checks; verify explicitly too
    assert folds and all(isinstance(f, tuple) and len(f) == 2 for f in folds)
    for tr, te in folds:
        assert tr.dtype == np.int64 and te.dtype == np.int64
        assert len(tr) and len(te)
        assert not np.intersect1d(tr, te).size
    assert verify_folds(ws, folds, name) == declared_level(name)
    rep = leak_report(ws, folds, name)
    assert rep["declared_level_ok"] is True
    assert rep["test_sets_disjoint"] is True
    if name in ("window_random", "record_random", "record_kfold", "subject_kfold", "subject_loso",
                "chronological_blocked"):
        assert rep["test_coverage"] == 1.0


def test_subject_protocols_pass_time_level_trivially(ws):
    """Disjoint subjects can never overlap in time."""
    for name in ("subject_loso", "subject_kfold"):
        assert_no_leak(ws, make_folds(ws, name), "time", gap_s=1e9)


# --------------------------------------------------------------------------- #
# Leaky folds must raise, for every level
# --------------------------------------------------------------------------- #
def test_window_leak(ws):
    n = len(ws)
    tr = np.arange(0, int(0.8 * n))
    te = np.arange(int(0.8 * n) - 7, n)                # 7 windows in both
    with pytest.raises(LeakError) as ei:
        assert_no_leak(ws, [(tr, te)], "window", protocol="hand-made")
    e = ei.value
    assert e.level == "window" and e.violated == "window" and e.n_shared == 7 and e.fold == 0
    assert "7 window index(es)" in str(e) and "hand-made" in str(e)
    assert e.counts["shared_windows"] == 7
    for level in ("record", "subject", "time"):     # stronger levels also fail, naming the window check first
        with pytest.raises(LeakError) as ei2:
            assert_no_leak(ws, [(tr, te)], level)
        assert ei2.value.violated == "window"


def test_record_leak(ws):
    folds = make_folds(ws, "window_random")           # clean at window level ...
    assert_no_leak(ws, folds, "window")
    with pytest.raises(LeakError) as ei:              # ... leaks records
        assert_no_leak(ws, folds, "record", protocol="window_random")
    e = ei.value
    assert e.violated == "record" and e.level == "record" and e.n_shared > 0
    assert "record(s) appear in both" in str(e) and e.examples
    assert all(ex in set(ws.record) for ex in e.examples)
    # explicit construction: one record split across train and test, everything else clean
    loso = make_folds(ws, "subject_loso")
    tr, te = loso[0]
    rec = ws.record[te[0]]
    leaky = (np.concatenate([tr, [te[0]]]), te[1:])
    with pytest.raises(LeakError) as ei:
        assert_no_leak(ws, [leaky], "record")
    assert ei.value.n_shared == 1 and ei.value.examples == (rec,)
    assert ei.value.counts["shared_record_test_windows"] == (ws.record[te[1:]] == rec).sum()


def test_subject_leak(ws):
    for name in ("record_random", "record_kfold"):
        folds = make_folds(ws, name)
        assert_no_leak(ws, folds, "record")
        with pytest.raises(LeakError) as ei:
            assert_no_leak(ws, folds, "subject", protocol=name)
        e = ei.value
        assert e.violated == "subject" and e.level == "subject" and e.n_shared > 0
        assert "subject(s) appear in both" in str(e)
        assert set(e.examples) <= set(ws.subject)
    # one subject's single test record moved into train of another fold
    loso = make_folds(ws, "subject_loso")
    tr, te = loso[0]
    held = ws.subject[te[0]]
    rec = ws.record[te[0]]
    move = te[ws.record[te] == rec]
    leaky = (np.sort(np.concatenate([tr, move])), te[~np.isin(te, move)])
    with pytest.raises(LeakError) as ei:
        assert_no_leak(ws, [leaky], "subject")
    assert ei.value.violated == "subject" and ei.value.examples == (held,)


def test_subject_level_also_catches_shared_records_with_disjoint_subjects():
    """A corrupted dataset: the same record id under two subjects."""
    w = make_ws(n_subjects=3, records_per_subject=4, windows_per_record=2, small_subject=False)
    rec = w.record.copy()
    rec[w.subject == "S01"] = np.where(rec[w.subject == "S01"] == "S01_R000", "S00_R000", rec[w.subject == "S01"])
    w2 = _windowset(X=w.X, y=w.y, subject=w.subject, record=rec, time=w.time, meta=w.meta, fs=64, source="corrupt")
    tr = np.flatnonzero(w2.subject != "S01")
    te = np.flatnonzero(w2.subject == "S01")
    assert not set(w2.subject[tr]) & set(w2.subject[te])
    with pytest.raises(LeakError) as ei:
        assert_no_leak(w2, [(tr, te)], "subject")
    assert ei.value.violated == "record" and ei.value.examples == ("S00_R000",)


def test_time_leak_guard_gap(ws):
    nogap = make_folds(ws, "chronological_nogap")
    assert_no_leak(ws, nogap, "time", gap_s=0.0)                      # honest at gap 0 ...
    assert_no_leak(ws, nogap, "time", gap_s=299.0)                    # ... and below the reading spacing
    with pytest.raises(LeakError) as ei:                              # ... but not with a 1-h guard gap
        assert_no_leak(ws, nogap, "time", gap_s=3600.0, protocol="chronological_nogap")
    e = ei.value
    assert e.violated == "time" and e.level == "time" and e.n_shared > 0
    assert "gap_s=3600" in str(e) and e.examples
    assert {"record", "subject", "t", "nearest_train_t", "distance_s"} <= set(e.examples[0])
    assert 0 < e.examples[0]["distance_s"] <= 3600.0
    assert e.counts["time_overlap_records"] > 0
    gap = make_folds(ws, "chronological")                             # built with the 1-h gap: passes at 1 h
    assert_no_leak(ws, gap, "time", gap_s=3600.0)
    with pytest.raises(LeakError):                                    # but a 2-h requirement fails
        assert_no_leak(ws, gap, "time", gap_s=7200.0)
    assert verify_folds(ws, make_folds(ws, "chronological", gap_s=7200.0), "chronological", gap_s=7200.0) == "time"


def test_time_leak_train_record_moved_into_test(ws):
    tr, te = make_folds(ws, "chronological")[0]
    s = "S00"
    rec = ws.record[tr[ws.subject[tr] == s]][0]                      # the subject's earliest training record
    move = tr[ws.record[tr] == rec]
    leaky = (tr[~np.isin(tr, move)], np.sort(np.concatenate([te, move])))
    with pytest.raises(LeakError) as ei:
        assert_no_leak(ws, [leaky], "time", gap_s=3600.0)
    e = ei.value
    assert e.violated == "time" and e.n_shared == len(move)
    assert e.examples[0]["record"] == rec and e.examples[0]["subject"] == s
    assert e.examples[0]["distance_s"] == pytest.approx(300.0)        # its neighbour reading is still in train
    assert_no_leak(ws, [leaky], "record")                             # records still disjoint: only time leaks


def test_time_leak_test_before_train_within_gap(ws):
    """Autocorrelation is symmetric: a test reading just BEFORE training also leaks."""
    t = window_time(ws)
    m = ws.subject == "S00"
    ut = np.unique(t[m])
    te = np.flatnonzero(m & (t == ut[0]))                             # first reading of S00
    tr = np.flatnonzero(m & (t >= ut[2]))                             # training starts two readings (600 s) later
    assert_no_leak(ws, [(tr, te)], "time", gap_s=0.0)
    assert_no_leak(ws, [(tr, te)], "time", gap_s=599.0)
    with pytest.raises(LeakError):
        assert_no_leak(ws, [(tr, te)], "time", gap_s=600.0)
    rep = leak_report(ws, [(tr, te)], gap_s=0.0)
    assert rep["folds"][0]["min_test_train_distance_s"] == pytest.approx(600.0)


def test_time_level_needs_times(ws_notime):
    w = _windowset(X=ws_notime.X, y=ws_notime.y, subject=ws_notime.subject, record=ws_notime.record, time=None,
                   meta=ws_notime.meta.drop(columns=["egv_time"]), fs=64, source="no-time")
    folds = make_folds(w, "subject_loso")
    with pytest.raises(ValueError, match="time"):
        assert_no_leak(w, folds, "time")
    with pytest.raises(ValueError, match="time"):
        make_folds(w, "chronological")
    rep = leak_report(w, folds, "subject_loso")
    assert rep["has_time"] is False and rep["passes"]["time"] is None and rep["declared_level_ok"] is True


def test_time_level_rejects_nan_times(ws):
    t = ws.time.copy(); t[3] = np.nan
    w = _windowset(X=ws.X, y=ws.y, subject=ws.subject, record=ws.record, time=t, meta=ws.meta, fs=64, source="nan")
    with pytest.raises(ValueError, match="NaN"):
        assert_no_leak(w, make_folds(w, "subject_loso"), "time")


def test_chronological_blocked_is_time_clean_and_sensitive_to_gap(ws):
    folds = make_folds(ws, "chronological_blocked", n_splits=4, gap_s=1800.0)
    assert len(folds) == 4
    assert_no_leak(ws, folds, "time", gap_s=1800.0)
    # readings sit on a 300-s grid and train needs t < lo - 1800, so the nearest training reading is 2100 s away
    assert leak_report(ws, folds, gap_s=1800.0)["totals"]["min_test_train_distance_s"] == pytest.approx(2100.0)
    assert_no_leak(ws, folds, "time", gap_s=2099.0)
    with pytest.raises(LeakError):
        assert_no_leak(ws, folds, "time", gap_s=2100.0)
    rep = leak_report(ws, folds, "chronological_blocked", gap_s=1800.0)
    assert rep["test_coverage"] == 1.0 and rep["test_sets_disjoint"]
    assert rep["totals"]["min_test_train_distance_s"] > 1800.0
    assert rep["totals"]["test_in_train_span_windows"] > 0            # interior blocks interpolate ...
    chrono = leak_report(ws, make_folds(ws, "chronological"), "chronological")
    assert chrono["totals"]["test_in_train_span_windows"] == 0        # ... chronological extrapolates
    # every reading of every subject tested exactly once, and train is on both sides of the block
    for k, (tr, te) in enumerate(folds):
        t = window_time(ws)
        for s in np.unique(ws.subject[te]):
            lo, hi = t[te][ws.subject[te] == s].min(), t[te][ws.subject[te] == s].max()
            ts = t[tr][ws.subject[tr] == s]
            assert ((ts < lo - 1800.0) | (ts > hi + 1800.0)).all()


def test_every_protocol_has_a_leaky_counterpart_that_fails_its_level(ws):
    """Shuffling the subject (or record / time) labels makes any split leaky."""
    for name, spec in PROTOCOLS.items():
        folds = make_folds(ws, name)
        tr, te = folds[0]
        if spec.level == "window":
            leaky = [(np.concatenate([tr, te[:3]]), te)]
            with pytest.raises(LeakError):
                assert_no_leak(ws, leaky, "window")
            continue
        # corrupt the keys of a few test windows so they collide with training keys
        take = te[:4]
        if spec.level == "record":
            rec = ws.record.copy(); rec[take] = ws.record[tr[0]]
            w = _windowset(X=ws.X, y=ws.y, subject=ws.subject, record=rec, time=ws.time, meta=ws.meta, fs=64, source="x")
        elif spec.level == "subject":
            sub = ws.subject.copy(); sub[take] = ws.subject[tr[0]]
            w = _windowset(X=ws.X, y=ws.y, subject=sub, record=ws.record, time=ws.time, meta=ws.meta, fs=64, source="x")
        else:
            t = ws.time.copy(); t[take] = ws.time[tr[ws.subject[tr] == ws.subject[take[0]]]][0]
            w = _windowset(X=ws.X, y=ws.y, subject=ws.subject, record=ws.record, time=t, meta=ws.meta, fs=64, source="x")
        with pytest.raises(LeakError) as ei:
            assert_no_leak(w, folds, spec.level, gap_s=spec.gap_for({}), protocol=name)
        assert ei.value.level == spec.level and ei.value.protocol == name


def test_unknown_level_is_a_value_error(ws):
    with pytest.raises(ValueError, match="level must be one of"):
        assert_no_leak(ws, make_folds(ws, "subject_loso"), "patient")


# --------------------------------------------------------------------------- #
# leak_report contents
# --------------------------------------------------------------------------- #
def test_leak_report_clean_loso(ws):
    folds = make_folds(ws, "subject_loso")
    rep = leak_report(ws, folds, "loso")
    assert rep["protocol"] == "subject_loso" and rep["declared_level"] == "subject"
    assert rep["n_folds"] == len(folds) == 5 and rep["n_windows"] == len(ws)
    assert rep["n_subjects"] == 5 and rep["n_records"] == len(np.unique(ws.record))
    assert rep["totals"]["shared_windows"] == rep["totals"]["shared_records"] == rep["totals"]["shared_subjects"] == 0
    assert rep["totals"]["time_overlap_windows"] == 0
    assert rep["passes"] == {"window": True, "record": True, "subject": True, "time": True}
    assert rep["test_coverage"] == 1.0 and rep["test_sets_disjoint"] and rep["declared_level_ok"]
    assert rep["fingerprint"] == fold_fingerprint(folds)
    for k, row in enumerate(rep["folds"]):
        assert row["fold"] == k and row["n_train"] + row["n_test"] == len(ws)
        assert row["test_subjects"] == 1 and row["train_subjects"] == 4
        assert row["min_test_train_distance_s"] is None            # held-out subject never seen in train
    json.dumps(rep)                                               # JSON-serialisable


def test_leak_report_window_random_counts_exactly(ws):
    folds = make_folds(ws, "window_random", seed=0)
    rep = leak_report(ws, folds, "window_random")
    assert rep["declared_level_ok"] is True and rep["passes"]["window"] is True
    assert rep["passes"]["record"] is False and rep["passes"]["subject"] is False and rep["passes"]["time"] is False
    for (tr, te), row in zip(folds, rep["folds"]):
        shared_r = set(ws.record[tr]) & set(ws.record[te])
        assert row["shared_records"] == len(shared_r)
        assert row["shared_record_test_windows"] == sum(r in shared_r for r in ws.record[te])
        assert row["shared_subjects"] == len(set(ws.subject[tr]) & set(ws.subject[te]))
        # gap 0: a test window leaks in time iff a training window shares its reading time == shares its record
        assert row["time_overlap_windows"] == row["shared_record_test_windows"]
        assert row["min_test_train_distance_s"] == 0.0
        assert row["shared_record_test_windows"] <= row["test_in_train_span_windows"] <= len(te)
    assert rep["totals"]["shared_records"] == sum(r["shared_records"] for r in rep["folds"])


def test_leak_report_overlapping_test_sets_and_partial_coverage(ws):
    tr, te = make_folds(ws, "chronological")[0]
    rep = leak_report(ws, [(tr, te), (tr, te)], "chronological")
    assert rep["test_sets_disjoint"] is False and rep["test_coverage"] < 1.0
    assert rep["declared_level_ok"] is True and rep["gap_s"] == 3600.0
    assert rep["totals"]["min_test_train_distance_s"] > 3600.0


# --------------------------------------------------------------------------- #
# Malformed folds, determinism, persistence
# --------------------------------------------------------------------------- #
def test_malformed_folds_are_value_errors(ws):
    n = len(ws)
    good = make_folds(ws, "subject_loso")[0]
    with pytest.raises(ValueError, match="out of range"):
        assert_no_leak(ws, [(np.array([0, 1, n]), np.array([2]))], "window")
    with pytest.raises(ValueError, match="duplicates"):
        assert_no_leak(ws, [(np.array([0, 1, 1]), np.array([2]))], "window")
    with pytest.raises(ValueError, match="empty test"):
        assert_no_leak(ws, [(good[0], np.array([], np.int64))], "window")
    with pytest.raises(ValueError, match="1-D integer"):
        assert_no_leak(ws, [(good[0].astype(float), good[1])], "window")
    mask_tr, mask_te = np.zeros(n, bool), np.zeros(n, bool)
    mask_tr[good[0]] = True; mask_te[good[1]] = True
    assert_no_leak(ws, [(mask_tr, mask_te)], "subject")               # boolean masks are accepted


def test_determinism_and_seed_sensitivity(ws):
    assert_same_folds(make_folds(ws, "window_random", seed=3), make_folds(ws, "window_random", seed=3))
    assert_same_folds(make_folds(ws, "record_random", seed=3), make_folds(ws, "record_random", seed=3))
    assert fold_fingerprint(make_folds(ws, "window_random", seed=0)) != fold_fingerprint(make_folds(ws, "window_random", seed=1))
    assert fold_fingerprint(make_folds(ws, "record_random", seed=0)) != fold_fingerprint(make_folds(ws, "record_random", seed=1))
    for name in ("subject_loso", "subject_kfold", "record_kfold", "chronological", "personal", "chronological_blocked"):
        assert fold_fingerprint(make_folds(ws, name, seed=0)) == fold_fingerprint(make_folds(ws, name, seed=9))
        assert PROTOCOLS[name].uses_seed is False


def test_make_folds_check_flag(ws, monkeypatch):
    """`check=False` skips the self-check; a broken builder is caught with the default."""
    def broken(w, seed=0):
        return [(np.arange(len(w)), np.arange(3))]
    monkeypatch.setitem(PROTOCOLS, "broken", P.ProtocolSpec("broken", broken, "window", "bad", {}, False))
    with pytest.raises(LeakError):
        make_folds(ws, "broken")
    assert len(make_folds(ws, "broken", check=False)) == 1


def test_folds_json_roundtrip(ws, tmp_path):
    folds = make_folds(ws, "personal")
    s = folds_to_json(folds, protocol="personal", seed=0)
    assert_same_folds(folds_from_json(s), folds)
    path = tmp_path / "folds.json"
    folds_to_json(folds, str(path), protocol="personal")
    assert_same_folds(folds_from_json(str(path)), folds)
    payload = json.loads(s)
    payload["folds"][0]["test"] = payload["folds"][0]["test"][1:]
    with pytest.raises(ValueError, match="fingerprint"):
        folds_from_json(json.dumps(payload))


def test_describe_folds(ws):
    txt = P.describe_folds(ws, make_folds(ws, "subject_loso"), "subject_loso")
    assert txt.startswith("subject_loso: 5 fold(s)") and txt.count("\n") == 5 and "S00" in txt
