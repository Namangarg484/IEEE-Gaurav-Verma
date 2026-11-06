"""
Tests for ppgbench.data: registry mechanics, the deterministic synthetic
dataset, and numeric parity of the Mendeley / BIG IDEAs loaders with the
legacy ``ppg_data.py`` on small fixtures written to ``tmp_path``.

Fast tests use only synthetic data and tiny fixtures (< 10 s).  Parity on the
real files is marked ``slow`` and runs only with ``PPGBENCH_SLOW=1``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ppgbench.data as pdata  # noqa: E402
from ppgbench.data import (  # noqa: E402
    DatasetError, bigideas, fingerprint, list_datasets, load_dataset, mendeley, register_dataset,
    synthetic, validate_windowset,
)
from ppgbench.data.registry import _REGISTRY, clear_cache, dataset_info  # noqa: E402

SLOW = os.environ.get("PPGBENCH_SLOW") == "1"
slow = pytest.mark.skipif(not SLOW, reason="set PPGBENCH_SLOW=1 for parity tests on the real files")


@pytest.fixture(scope="module")
def ws():
    return load_dataset("synthetic", n_subjects=4, records_per_subject=6, windows_per_record=3, seed=0)


@pytest.fixture
def legacy():
    """The reference implementation; skipped when its heavy deps are absent."""
    return pytest.importorskip("ppg_data")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_builtin_datasets_registered():
    assert {"mendeley", "bigideas", "synthetic"} <= set(list_datasets())
    assert list_datasets() == sorted(list_datasets())


def test_dataset_info_and_params():
    info = dataset_info("synthetic")
    assert info.requires_files is False and info.is_available()
    assert {"n_subjects", "records_per_subject", "windows_per_record", "seed"} <= set(info.params)
    assert "synthetic" in pdata.describe_datasets()
    for name in ("mendeley", "bigideas"):
        assert isinstance(pdata.is_available(name), bool)


def test_unknown_dataset_lists_alternatives():
    with pytest.raises(KeyError, match="synthetic"):
        load_dataset("no-such-dataset")


def test_unknown_kwarg_is_a_typeerror():
    with pytest.raises(TypeError, match="n_subjects"):
        load_dataset("synthetic", n_subjcts=3)


def test_register_duplicate_and_overwrite():
    @register_dataset("_tmp_dummy", requires_files=False)
    def _dummy():
        return load_dataset("synthetic", n_subjects=1, records_per_subject=1, windows_per_record=1)

    try:
        with pytest.raises(ValueError, match="already registered"):
            register_dataset("_tmp_dummy")(_dummy)
        register_dataset("_tmp_dummy", requires_files=False, overwrite=True)(_dummy)
        assert _dummy.dataset_name == "_tmp_dummy"
        assert len(load_dataset("_tmp_dummy")) == 1
    finally:
        _REGISTRY.pop("_tmp_dummy", None)


def test_loader_must_return_windowset():
    @register_dataset("_tmp_bad", requires_files=False)
    def _bad():
        return 42

    try:
        with pytest.raises(DatasetError, match="expected WindowSet"):
            load_dataset("_tmp_bad")
    finally:
        _REGISTRY.pop("_tmp_bad", None)


def test_cache_returns_same_object():
    clear_cache()
    a = load_dataset("synthetic", n_subjects=2, records_per_subject=2, windows_per_record=1, cache=True)
    b = load_dataset("synthetic", n_subjects=2, records_per_subject=2, windows_per_record=1, cache=True)
    c = load_dataset("synthetic", n_subjects=2, records_per_subject=2, windows_per_record=1, cache=False)
    assert a is b and a is not c
    clear_cache()


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
def test_synthetic_shapes_and_dtypes(ws):
    n = 4 * 6 * 3
    assert ws.X.shape == (n, 320) and ws.X.dtype == np.float32
    assert ws.y.shape == (n,) and ws.y.dtype == np.float32
    assert ws.subject.shape == (n,) and ws.subject.dtype.kind in "UO"
    assert ws.record.shape == (n,) and ws.record.dtype.kind in "UO"
    assert ws.time is not None and ws.time.shape == (n,) and ws.time.dtype == np.float64
    assert ws.fs == 64 and ws.source == "synthetic" and len(ws) == n
    assert np.isfinite(ws.X).all() and (ws.y >= 40).all() and (ws.y <= 400).all()
    assert len(np.unique(ws.subject)) == 4 and len(np.unique(ws.record)) == 24


def test_synthetic_is_zscored_per_window(ws):
    assert np.abs(ws.X.mean(1)).max() < 1e-4
    assert np.abs(ws.X.std(1) - 1).max() < 1e-3


def test_synthetic_deterministic():
    a = load_dataset("synthetic", n_subjects=3, records_per_subject=4, windows_per_record=2, seed=7)
    b = load_dataset("synthetic", n_subjects=3, records_per_subject=4, windows_per_record=2, seed=7)
    c = load_dataset("synthetic", n_subjects=3, records_per_subject=4, windows_per_record=2, seed=8)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and np.array_equal(a.time, b.time)
    pd.testing.assert_frame_equal(a.meta, b.meta)
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)
    assert not np.array_equal(a.y, c.y)


def test_synthetic_subject_subset_stability():
    """Subject k is identical whether 3 or 5 subjects are requested."""
    small = load_dataset("synthetic", n_subjects=3, records_per_subject=5, windows_per_record=2, seed=0)
    big = load_dataset("synthetic", n_subjects=5, records_per_subject=5, windows_per_record=2, seed=0)
    m = np.isin(big.subject, np.unique(small.subject))
    assert np.array_equal(small.X, big.X[m]) and np.array_equal(small.y, big.y[m])
    assert np.array_equal(small.time, big.time[m])


def test_synthetic_time_is_monotone_with_300s_spacing(ws):
    for s in np.unique(ws.subject):
        t = ws.time[ws.subject == s]
        assert np.all(np.diff(t) >= 0), "time must be non-decreasing in window order"
        u = np.unique(t)
        assert len(u) == 6 and np.allclose(np.diff(u), 300.0)
    starts = [ws.time[ws.subject == s].min() for s in np.unique(ws.subject)]
    assert len(set(starts)) == 4, "subjects start at distinct times"
    # one timestamp per record and the meta copy agrees with the array
    df = pd.DataFrame({"record": ws.record, "t": ws.time})
    assert (df.groupby("record").t.nunique() == 1).all()
    mt = ws.meta.set_index("record").egv_time
    assert np.allclose(mt.loc[ws.record].to_numpy(), ws.time)


def test_synthetic_meta_consistent(ws):
    meta = ws.meta
    assert {"record", "subject", "glucose", "egv_time", "hr_bpm", "age", "sex", "height", "weight",
            "n_windows"} <= set(meta.columns)
    assert len(meta) == 24 and meta.record.is_unique
    assert set(meta.record) == set(np.unique(ws.record))
    glu = meta.set_index("record").glucose
    assert np.allclose(glu.loc[ws.record].to_numpy(np.float32), ws.y)
    assert meta.n_windows.sum() == len(ws) and (meta.n_windows == 3).all()
    assert set(meta.sex) <= {"Male", "Female"}
    assert validate_windowset(ws) == []


def test_synthetic_target_varies_within_subject(ws):
    within = ws.meta.groupby("subject").glucose.std()
    assert (within > 0).all(), "chronological protocols need within-subject variation"
    between = ws.meta.groupby("subject").glucose.mean().std()
    assert between > within.mean(), "subject offsets dominate, as in real data"


def test_synthetic_signal_dependent_term_is_learnable():
    """Dominant frequency (heart rate) read from the window predicts glucose."""
    ws = load_dataset("synthetic", n_subjects=5, records_per_subject=12, windows_per_record=2, seed=1)
    f = np.fft.rfftfreq(ws.X.shape[1], 1 / ws.fs)
    P = np.abs(np.fft.rfft(ws.X, axis=1)) ** 2
    band = (f >= 0.7) & (f <= 3.0)
    hr = f[band][P[:, band].argmax(1)] * 60
    df = pd.DataFrame({"record": ws.record, "hr": hr, "y": ws.y}).groupby("record").mean()
    meta = ws.meta.set_index("record").loc[df.index]
    assert np.corrcoef(df.hr, meta.hr_bpm)[0, 1] > 0.9, "estimated HR must match the generating HR"
    # within-subject: the HR term is the only non-noise component of the target
    rs = [np.corrcoef(g.hr, g.y)[0, 1] for _, g in df.join(meta.subject).groupby("subject")]
    assert np.median(rs) > 0.5


def test_synthetic_demographics(ws):
    if hasattr(ws, "demographics"):
        d = ws.demographics()
        assert d is not None and d.shape == (len(ws), 4) and np.isfinite(d).all()


def test_synthetic_rejects_bad_sizes():
    with pytest.raises(DatasetError):
        load_dataset("synthetic", n_subjects=0)
    with pytest.raises(DatasetError):
        load_dataset("synthetic", windows_per_record=0)


def test_synthetic_ppg_has_requested_rate():
    rng = np.random.default_rng(0)
    x = synthetic.synthetic_ppg(64 * 20, 64, 84.0, rng)
    f = np.fft.rfftfreq(len(x), 1 / 64)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    band = (f >= 0.7) & (f <= 3.0)
    assert abs(f[band][P[band].argmax()] * 60 - 84.0) < 4.0


def test_synthetic_defaults_support_chronological_protocols():
    """The default synthetic set must be usable by the time-based protocols (spec: data).

    A subject spans 48 x 300 s = 4 h, so the 1-h guard gap of ``chronological``
    leaves a non-empty train set and ``personal`` finds > 64 training windows
    per subject.  Integration test against the binding ``make_folds`` interface.
    """
    protocols = pytest.importorskip("ppgbench.protocols")
    ws = load_dataset("synthetic", seed=0)
    assert len(np.unique(ws.subject)) == 6 and ws.meta.groupby("subject").size().eq(48).all()
    span = ws.meta.groupby("subject").egv_time.agg(lambda t: t.max() - t.min())
    assert (span == 47 * 300.0).all()
    for name in ("chronological", "chronological_nogap", "personal"):
        folds = protocols.make_folds(ws, name, seed=0)
        assert folds, f"{name}: no folds on the default synthetic set"
        for tr, te in folds:
            assert len(tr) and len(te), f"{name}: empty train or test set"
            assert ws.time[tr].max() < ws.time[te].min() or name != "personal"
        protocols.assert_no_leak(ws, folds, "time")
    personal = protocols.make_folds(ws, "personal", seed=0)
    assert len(personal) == 6 and all(len(tr) > 64 for tr, _ in personal)


# --------------------------------------------------------------------------- #
# Validation and fingerprint
# --------------------------------------------------------------------------- #
def _clone(ws):
    import copy
    return copy.copy(ws)


def test_validate_catches_length_mismatch(ws):
    bad = _clone(ws)
    bad.y = ws.y[:-1]
    with pytest.raises(DatasetError, match="len"):
        validate_windowset(bad)
    assert validate_windowset(bad, strict=False)


def test_validate_catches_record_subject_conflict(ws):
    bad = _clone(ws)
    subj = ws.subject.copy()
    subj[0] = "ZZ"
    bad.subject = subj
    with pytest.raises(DatasetError, match="more than one subject"):
        validate_windowset(bad)


def test_validate_catches_meta_mismatch(ws):
    bad = _clone(ws)
    bad.meta = ws.meta.iloc[1:].reset_index(drop=True)
    with pytest.raises(DatasetError, match="meta records"):
        validate_windowset(bad)
    bad = _clone(ws)
    m = ws.meta.copy()
    m.loc[0, "glucose"] += 50
    bad.meta = m
    with pytest.raises(DatasetError, match="glucose"):
        validate_windowset(bad)


def test_validate_catches_nan_and_time_problems(ws):
    bad = _clone(ws)
    X = ws.X.copy()
    X[0, 0] = np.nan
    bad.X = X
    with pytest.raises(DatasetError, match="non-finite"):
        validate_windowset(bad)
    bad = _clone(ws)
    t = ws.time.copy()
    t[0] += 1
    bad.time = t
    with pytest.raises(DatasetError, match="timestamp"):
        validate_windowset(bad)


def test_fingerprint_is_sensitive(ws):
    a = fingerprint(ws)
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)
    mod = _clone(ws)
    y = ws.y.copy()
    y[0] += 1
    mod.y = y
    assert fingerprint(mod) != a
    mod = _clone(ws)
    mod.time = None
    assert fingerprint(mod) != a


# --------------------------------------------------------------------------- #
# Mendeley loader on a tiny fixture (parity with ppg_data.load_mendeley)
# --------------------------------------------------------------------------- #
def _write_mendeley_fixture(root: Path, records: dict[str, dict]) -> None:
    (root / "RawData").mkdir(parents=True)
    (root / "Labels").mkdir()
    rng = np.random.default_rng(123)
    for key, lab in records.items():
        n = lab.get("n", mendeley.RECORD_SAMPLES)
        t = np.arange(n) / mendeley.MENDELEY_FS
        x = 500 + 40 * np.sin(2 * np.pi * lab["hr"] / 60 * t) + rng.normal(0, 2, n)
        pd.DataFrame(np.round(x).astype(int)).to_csv(root / "RawData" / f"signal_{key}.csv",
                                                     header=False, index=False)
        pd.DataFrame([{"ID": int(key.split("_")[0]), "Gender": lab["sex"], "Age": lab["age"],
                       "Glucose": lab["glucose"], "Height": 170, "Weight": 65}]).to_csv(
            root / "Labels" / f"label_{key}.csv", index=False)


@pytest.fixture
def mendeley_root(tmp_path):
    root = tmp_path / "PPG_Dataset"
    _write_mendeley_fixture(root, {
        "01_0001": {"hr": 72, "sex": "Male", "age": 38, "glucose": 99},
        "01_0002": {"hr": 80, "sex": "Male", "age": 38, "glucose": 104},
        "02_0001": {"hr": 65, "sex": "Female", "age": 51, "glucose": 143},
    })
    (root / "RawData" / "signal_03_0001.csv").write_text("1\n2\n3\n")   # no label -> skipped
    return root


def test_mendeley_loader_matches_legacy(mendeley_root, legacy):
    ours = load_dataset("mendeley", root=str(mendeley_root))
    ref = legacy.load_mendeley(root=str(mendeley_root))
    assert ours.X.shape == ref.X.shape == (9, 320)
    np.testing.assert_allclose(ours.X, ref.X, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ours.y, ref.y)
    np.testing.assert_array_equal(ours.subject, ref.subject)
    np.testing.assert_array_equal(ours.record, ref.record)
    assert ours.time is None and ours.source == "Mendeley" and ours.fs == 64
    cols = ["record", "subject", "glucose", "gender", "age", "height", "weight", "raw_std", "n_windows"]
    pd.testing.assert_frame_equal(ours.meta[cols], ref.meta[cols])
    assert (ours.meta.sex == ours.meta.gender).all()
    assert validate_windowset(ours) == []


def test_mendeley_subject_filter_and_helpers(mendeley_root):
    assert mendeley.list_records(str(mendeley_root)) == ["01_0001", "01_0002", "02_0001"]
    assert mendeley.is_available(str(mendeley_root))
    ws = load_dataset("mendeley", root=str(mendeley_root), subjects=["02"])
    assert set(ws.subject) == {"02"} and len(ws) == 3
    raw, lab = mendeley.read_record(str(mendeley_root), "02_0001")
    assert raw.shape == (mendeley.RECORD_SAMPLES,) and lab["glucose"] == 143 and lab["sex"] == "Female"
    with pytest.raises(FileNotFoundError):
        load_dataset("mendeley", root=str(mendeley_root / "nowhere"))
    with pytest.raises(DatasetError):
        load_dataset("mendeley", root=str(mendeley_root), subjects=["99"])


# --------------------------------------------------------------------------- #
# BIG IDEAs loader on a tiny fixture (parity with ppg_data.load_bigideas)
# --------------------------------------------------------------------------- #
def _write_bigideas_fixture(d: Path, sid: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    L = 7680
    t = 1.58e9 + 300.0 * np.arange(n)
    glucose = rng.uniform(60, 220, n).astype(np.float32)
    glucose[1] = 30.0                     # below reportable range -> dropped
    glucose[2] = 450.0                    # above -> dropped
    tt = np.arange(L) / 64.0
    bvp = (np.sin(2 * np.pi * 1.2 * tt)[None] * rng.uniform(20, 80, (n, 1))
           + rng.normal(0, 3, (n, L))).astype(np.float32)
    bvp[3, :] = np.nan                    # device off -> no windows
    bvp[4, 3840 - 100:3840 - 50] = np.nan  # gap inside the last pre-reading window only
    bvp[5, 3840 - 640:3840 - 320] = 0.0   # flat window
    np.savez(d / f"sub_{sid}.npz", egv_time=t, glucose=glucose, roc=np.full(n, np.nan, np.float32),
             bvp=bvp, bvp_coverage=np.isfinite(bvp).mean(1).astype(np.float32))


@pytest.fixture
def bigideas_dir(tmp_path):
    _write_bigideas_fixture(tmp_path, "001", 8, 1)
    _write_bigideas_fixture(tmp_path, "002", 6, 2)
    return tmp_path


@pytest.mark.parametrize("max_per_egv", [2, 12])
def test_bigideas_loader_matches_legacy(bigideas_dir, legacy, max_per_egv):
    ours = load_dataset("bigideas", dir_path=str(bigideas_dir), max_windows_per_egv=max_per_egv)
    ref = legacy.load_bigideas(dir_path=str(bigideas_dir), max_windows_per_egv=max_per_egv)
    assert ours.X.shape == ref.X.shape and len(ours) > 0
    np.testing.assert_allclose(ours.X, ref.X, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ours.y, ref.y)
    np.testing.assert_array_equal(ours.subject, ref.subject)
    np.testing.assert_array_equal(ours.record, ref.record)
    cols = ["record", "subject", "glucose", "egv_time", "n_windows"]
    pd.testing.assert_frame_equal(ours.meta[cols], ref.meta[cols])
    assert ours.time is not None and ours.time.dtype == np.float64
    rec_t = ref.meta.set_index("record").egv_time
    np.testing.assert_array_equal(ours.time, rec_t.loc[ours.record].to_numpy())
    assert validate_windowset(ours) == []


def test_bigideas_window_selection_rules(bigideas_dir):
    ws = load_dataset("bigideas", dir_path=str(bigideas_dir), subjects=[1])
    recs = ws.meta.set_index("record")
    assert "001_00001" not in recs.index and "001_00002" not in recs.index   # out-of-range glucose
    assert "001_00003" not in recs.index                                     # all-NaN BVP
    assert recs.loc["001_00004", "n_windows"] == 11                          # one window with a gap
    assert recs.loc["001_00005", "n_windows"] == 11                          # one flat window
    assert recs.loc["001_00000", "n_windows"] == 12
    assert (ws.time[ws.record == "001_00000"] == 1.58e9).all()
    ws2 = load_dataset("bigideas", dir_path=str(bigideas_dir), subjects=["001"], lookback_s=20.0)
    assert ws2.meta.n_windows.max() == 4
    assert bigideas.list_subjects(str(bigideas_dir)) == ["001", "002"]
    assert bigideas.normalise_subjects([1, "2", "003"]) == ["001", "002", "003"]
    with pytest.raises(FileNotFoundError):
        load_dataset("bigideas", dir_path=str(bigideas_dir), subjects=["999"])


def test_bigideas_read_subject_validates(tmp_path):
    np.savez(tmp_path / "sub_009.npz", egv_time=np.arange(3.0), glucose=np.ones(3, np.float32))
    with pytest.raises(DatasetError, match="missing arrays"):
        bigideas.read_subject(str(tmp_path / "sub_009.npz"))


def test_fetch_wrapper_argument_check(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        bigideas.fetch_bigideas(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        bigideas.fetch_bigideas(str(tmp_path), dir_path=str(tmp_path), script=str(tmp_path / "missing.py"))


# --------------------------------------------------------------------------- #
# Parity with the legacy loaders on the real files (slow)
# --------------------------------------------------------------------------- #
@slow
def test_real_mendeley_parity(legacy):
    if not mendeley.is_available():
        pytest.skip("Mendeley raw files not present")
    ours, ref = load_dataset("mendeley"), legacy.load_mendeley()
    assert ours.X.shape == ref.X.shape == (len(ref), 320)
    np.testing.assert_allclose(ours.X, ref.X, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ours.y, ref.y)
    np.testing.assert_array_equal(ours.record, ref.record)
    assert len(np.unique(ours.record)) == 67 and len(np.unique(ours.subject)) == 23


@slow
def test_real_bigideas_parity(legacy):
    subs = bigideas.list_subjects()
    if not subs:
        pytest.skip("BIG IDEAs npz files not present")
    ours = load_dataset("bigideas", subjects=subs[:1], max_windows_per_egv=2)
    ref = legacy.load_bigideas(subjects=subs[:1], max_windows_per_egv=2)
    np.testing.assert_allclose(ours.X, ref.X, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(ours.y, ref.y)
    np.testing.assert_array_equal(ours.record, ref.record)
    np.testing.assert_array_equal(ours.time, ref.meta.set_index("record").egv_time.loc[ours.record].to_numpy())
