"""
Tests for ``ppgbench.core``: numeric parity with the legacy ``ppg_data.py``
on synthetic signals, plus behaviour of the WindowSet container, the
signal-quality indices and the optional morphology features.

Run:  ~/.venvs/ppgfusion/bin/python -m pytest tests/ppgbench/test_core.py -q
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import ppgbench
from ppgbench.core import features as F
from ppgbench.core import signal as S
from ppgbench.core import views as V
from ppgbench.core.windowset import WindowSet, concat_windowsets

from .conftest import synthetic_ppg

legacy = pytest.importorskip("ppg_data", reason="legacy reference module ppg_data.py must be importable from ROOT")


# --------------------------------------------------------------------------- #
# package surface
# --------------------------------------------------------------------------- #
class TestPackage:
    def test_version(self):
        assert ppgbench.__version__ == "0.1.0"

    def test_import_is_lazy(self, root):
        """``import ppgbench`` must not drag torch/cv2 or the heavy sub-modules in."""
        code = ("import sys, ppgbench; "
                "heavy = [m for m in ('torch', 'cv2', 'ppgbench.train', 'ppgbench.report', 'ppgbench.models') "
                "if m in sys.modules]; print(','.join(heavy))")
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "", f"eager imports: {out.stdout.strip()}"
        for name in ("load_dataset", "make_folds", "build_model", "train_net", "summarize", "ResultStore"):
            assert name in ppgbench.__all__
            assert name in dir(ppgbench)

    def test_lazy_getattr_resolves_core_and_rejects_unknown(self):
        assert ppgbench.WindowSet is WindowSet
        assert ppgbench.handcrafted_features is F.handcrafted_features
        with pytest.raises(AttributeError):
            _ = ppgbench.definitely_not_a_symbol
        assert not hasattr(ppgbench, "definitely_not_a_symbol")

    @pytest.mark.parametrize("name,module", [
        ("load_dataset", "ppgbench.data.registry"), ("make_folds", "ppgbench.protocols"),
        ("build_model", "ppgbench.models.registry"), ("train_net", "ppgbench.train"),
        ("summarize", "ppgbench.metrics"), ("ResultStore", "ppgbench.report"),
    ])
    def test_lazy_exports_map_to_declared_modules(self, name, module):
        assert ppgbench._LAZY[name][0] == module
        if importlib.util.find_spec(module) is None:  # other agents' modules may not exist yet
            pytest.skip(f"{module} not implemented yet")
        obj = getattr(ppgbench, name)
        assert obj is getattr(importlib.import_module(module), name)

    def test_constants_match_legacy(self):
        assert (S.FS, S.WIN, S.HOP, S.MENDELEY_FS) == (legacy.FS, legacy.WIN, legacy.HOP, legacy.MENDELEY_FS)
        assert (V.TAU, V.PHASE_POINTS, V.CWT_SCALES, V.SPEC) == (legacy.TAU, legacy.PHASE_POINTS,
                                                                  legacy.CWT_SCALES, legacy.SPEC)


# --------------------------------------------------------------------------- #
# signal conditioning: numeric parity with ppg_data.py
# --------------------------------------------------------------------------- #
class TestSignalParity:
    def test_bandpass_identical(self, raw_mendeley_like):
        got = S.bandpass(raw_mendeley_like, 2175)
        ref = legacy.bandpass(raw_mendeley_like, 2175)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, ref)

    def test_bandpass_removes_dc_and_hf(self):
        fs = 64
        t = np.arange(fs * 20) / fs
        x = 5.0 + np.sin(2 * np.pi * 1.2 * t) + np.sin(2 * np.pi * 20.0 * t)
        y = S.bandpass(x, fs)
        assert abs(y.mean()) < 1e-2
        f, P = __import__("scipy.signal", fromlist=["welch"]).welch(y, fs=fs, nperseg=512)
        assert P[np.argmin(abs(f - 1.2))] > 100 * P[np.argmin(abs(f - 20.0))]

    def test_bandpass_rejects_bad_edges(self):
        with pytest.raises(ValueError):
            S.bandpass(np.zeros(100), 10, lo=0.5, hi=8.0)      # hi >= Nyquist
        with pytest.raises(ValueError):
            S.bandpass(np.zeros((2, 50)), 64)

    def test_to_rate_identical_to_legacy(self, raw_mendeley_like):
        got = S.to_rate(raw_mendeley_like, 2175)
        ref = legacy.to_common_rate(raw_mendeley_like, 2175)
        assert got.dtype == np.float32
        assert got.shape == ref.shape == (int(np.ceil(21_900 * 64 / 2175)),)
        np.testing.assert_array_equal(got, ref)

    def test_to_rate_native_rate_passthrough(self):
        x = synthetic_ppg(640, 64.0)
        y = S.to_rate(x, 64, 64)
        assert y.shape == x.shape and y.dtype == np.float32
        np.testing.assert_allclose(y, S.bandpass(x - x.mean(), 64).astype(np.float32))

    def test_to_rate_requires_integer_rates(self):
        with pytest.raises(ValueError):
            S.to_rate(np.zeros(1000), 2175.5)

    @pytest.mark.parametrize("n,win,hop", [(21900 * 64 // 2175, 320, 160), (320, 320, 160), (319, 320, 160),
                                           (1000, 320, 320), (0, 320, 160)])
    def test_sliding_windows_identical(self, n, win, hop):
        x = np.arange(n, dtype=np.float32)
        got, ref = S.sliding_windows(x, win, hop), legacy.sliding_windows(x, win, hop)
        assert got.shape == ref.shape
        np.testing.assert_array_equal(got, ref)
        if n >= win:
            assert got.shape[0] == 1 + (n - win) // hop
            np.testing.assert_array_equal(got[-1], x[(got.shape[0] - 1) * hop:(got.shape[0] - 1) * hop + win])
        else:
            assert got.shape == (0, win) and got.dtype == np.float32

    def test_sliding_windows_validation(self):
        with pytest.raises(ValueError):
            S.sliding_windows(np.zeros((3, 4)))
        with pytest.raises(ValueError):
            S.sliding_windows(np.zeros(10), win=4, hop=0)

    def test_zscore_rows_identical(self, raw_mendeley_like):
        W = S.sliding_windows(S.to_rate(raw_mendeley_like, 2175))
        got, ref = S.zscore_rows(W), legacy._zscore_rows(W)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, ref)
        np.testing.assert_allclose(got.mean(1), 0, atol=1e-5)
        np.testing.assert_allclose(got.std(1), 1, atol=1e-4)

    def test_zscore_flat_row_is_zero_not_nan(self):
        Z = S.zscore_rows(np.ones((2, 50)))
        assert np.isfinite(Z).all() and np.all(Z == 0)


class TestSQI:
    def test_keys_and_shapes(self, windows64):
        q = S.sqi(windows64, S.FS)
        assert set(q) == set(S.SQI_KEYS)
        for v in q.values():
            assert v.shape == (len(windows64),)
        assert q["hr_plausible"].dtype == bool and q["finite"].dtype == bool

    def test_clean_pulse_is_plausible_flatline_is_not(self):
        fs = 64
        t = np.arange(320) / fs
        good = np.sin(2 * np.pi * 1.3 * t) + 0.3 * np.sin(2 * np.pi * 2.6 * t)
        slow = np.sin(2 * np.pi * 0.2 * t)                     # 12 bpm: implausible
        flat = np.r_[np.sin(2 * np.pi * 1.3 * t[:100]), np.zeros(220)]   # 70 % flat
        X = S.zscore_rows(np.stack([good, slow, flat]))
        q = S.sqi(X, fs)
        assert q["hr_plausible"].tolist() == [True, False, False] or q["hr_plausible"][0]
        assert q["hr_plausible"][0] and not q["hr_plausible"][1]
        assert abs(q["dominant_freq_hz"][0] - 1.3) <= 0.25        # Welch bin width at nperseg 256
        assert q["flatline_frac"][2] > 0.6 > q["flatline_frac"][0]
        assert q["hr_band_power_frac"][0] > 0.9
        keep = S.sqi_mask(q)
        assert keep.tolist() == [True, False, False]

    def test_nonfinite_and_empty(self):
        X = np.zeros((2, 320))
        X[1, 5] = np.nan
        q = S.sqi(X)
        assert q["finite"].tolist() == [True, False]
        assert not q["hr_plausible"].any()                        # zero std / nonfinite never plausible
        q0 = S.sqi(np.empty((0, 320)))
        assert all(v.shape == (0,) for v in q0.values())
        assert S.sqi_summary(q0) == {"n": 0}
        assert S.sqi_summary(q)["n"] == 2


# --------------------------------------------------------------------------- #
# views: numeric parity
# --------------------------------------------------------------------------- #
class TestViewsParity:
    def test_cwt_identical(self, windows64):
        for x in windows64[:4]:
            got, ref = V.cwt_scalogram(x), legacy.cwt_scalogram(x)
            assert got.shape == (64, 64) and got.dtype == np.float32
            np.testing.assert_array_equal(got, ref)
            assert got.min() == 0.0 and got.max() == 1.0

    def test_cwt_constant_window_not_normalised(self):
        x = np.zeros(320, np.float32)
        out = V.cwt_scalogram(x)
        assert out.shape == (64, 64) and np.all(out == 0)

    def test_phase_space_identical_both_branches(self, windows64):
        x = windows64[0]
        np.testing.assert_array_equal(V.phase_space(x), legacy.phase_space(x))
        short = x[:100]                                           # 90 points < 256 -> zero padded
        got, ref = V.phase_space(short), legacy.phase_space(short)
        assert got.shape == (3, 256)
        np.testing.assert_array_equal(got, ref)
        assert np.all(got[:, 90:] == 0)

    def test_phase_space_embedding_rows(self):
        x = np.arange(300, dtype=np.float32)
        p = V.phase_space(x, tau=5, n_points=290)                 # exactly L points, no sub-sampling
        np.testing.assert_array_equal(p[0], x[:290])
        np.testing.assert_array_equal(p[1], x[5:295])
        np.testing.assert_array_equal(p[2], x[10:300])
        assert V.phase_space(np.zeros(8), tau=5).shape == (3, 256)

    def test_make_views_identical(self, windows64):
        torch = pytest.importorskip("torch")
        got, ref = V.make_views(windows64), legacy.make_views(windows64)
        assert set(got) == set(V.VIEW_KEYS) == set(ref)
        n = len(windows64)
        assert tuple(got["x1"].shape) == (n, 1, 320)
        assert tuple(got["x2"].shape) == (n, 1, 64, 64)
        assert tuple(got["x3"].shape) == (n, 3, 256)
        for k in V.VIEW_KEYS:
            assert got[k].dtype == torch.float32
            assert torch.equal(got[k], ref[k])
        shapes = V.view_shapes(320)
        for k in V.VIEW_KEYS:
            assert tuple(got[k].shape[1:]) == shapes[k]

    def test_make_views_numpy_and_empty(self):
        v = V.make_views_numpy(np.empty((0, 320), np.float32))
        assert v["x1"].shape == (0, 1, 320) and v["x2"].shape == (0, 1, 64, 64) and v["x3"].shape == (0, 3, 256)
        with pytest.raises(ValueError):
            V.make_views_numpy(np.zeros(320))


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
class TestFeatures:
    def test_handcrafted_identical(self, windows64):
        got, ref = F.handcrafted_features(windows64), legacy.handcrafted_features(windows64)
        assert got.shape == (len(windows64), 9) == (len(windows64), len(F.FEATURE_NAMES))
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, ref)

    def test_handcrafted_hr_tracks_synthetic_rate(self, windows64):
        hr = F.handcrafted_features(windows64)[:, 0]
        # windows come in groups of 3 with fundamental 0.9, 1.2, 1.6, 2.1 Hz
        expected = np.repeat([0.9, 1.2, 1.6, 2.1], 3) * 60
        assert np.all(np.abs(hr - expected) <= 0.25 * 60)        # one Welch bin (0.25 Hz)

    def test_handcrafted_finite_and_empty(self):
        out = F.handcrafted_features(np.zeros((3, 320), np.float32))
        assert np.isfinite(out).all() and out.shape == (3, 9)
        assert F.handcrafted_features(np.empty((0, 320))).shape == (0, 9)

    def test_morphology_features(self):
        pytest.importorskip("neurokit2")
        fs, hr = S.FS, 0.9
        t = np.arange(320) / fs
        # train of asymmetric Gaussian pulses (fast rise, slow decay) with a small dicrotic bump
        x = np.zeros_like(t)
        for k in range(-1, 7):
            t0 = k / hr + 0.1
            dt = t - t0
            x += np.where(dt < 0, np.exp(-(dt / 0.08) ** 2), np.exp(-(dt / 0.25) ** 2))
            x += 0.25 * np.exp(-((dt - 0.38) / 0.06) ** 2)
        X = S.zscore_rows(np.stack([x, np.roll(x, 40)]))
        m = F.morphology_features(X, fs=fs)
        assert m.shape == (2, len(F.MORPHOLOGY_FEATURE_NAMES)) and m.dtype == np.float32
        assert np.isfinite(m).all()
        assert 3 <= m[0, 0] <= 6                                  # 5 s at 0.9 Hz -> 4-5 systolic peaks
        assert abs(m[0, 1] - 1 / hr) < 0.1                        # mean IBI ~1.11 s
        assert m[0, 2] < 0.05 and m[0, 3] < 0.05                  # regular rhythm
        assert 0 < m[0, 4] < m[0, 5]                              # rise faster than decay
        assert 0 < m[0, 6] < m[0, 1]                              # half-width shorter than the beat
        assert m[0, 7] > 0 and 0 <= m[0, 8] < 0.5
        assert m[0, 9] > 0.5 and 0 < m[0, 10] < 1 and 0 < m[0, 11] < 1   # notch found in most beats
        assert 0 < m[0, 13] < 1
        nan_out = F.morphology_features(np.zeros((1, 320)), return_nan=True)
        assert np.isnan(nan_out[0, 1])                            # degenerate window -> NaN, not crash
        assert F.morphology_features(np.zeros((1, 320)))[0, 1] == 0.0

    def test_morphology_import_error_message(self, monkeypatch):
        monkeypatch.setattr(F, "neurokit_available", lambda: False)
        with pytest.raises(ImportError, match="neurokit2"):
            F.morphology_features(np.zeros((1, 320)))


# --------------------------------------------------------------------------- #
# WindowSet
# --------------------------------------------------------------------------- #
def _toy_ws(with_time: bool = True, with_demo: bool = True) -> WindowSet:
    rng = np.random.default_rng(0)
    rows, meta = [], []
    subj = np.array(["s1"] * 4 + ["s2"] * 2)
    rec = np.array(["s1_a", "s1_a", "s1_b", "s1_b", "s2_a", "s2_a"])
    y = np.array([100, 100, 150, 150, 90, 90], np.float32)
    for r, s, g in [("s1_a", "s1", 100.0), ("s1_b", "s1", 150.0), ("s2_a", "s2", 90.0)]:
        row = {"record": r, "subject": s, "glucose": g, "egv_time": {"s1_a": 0.0, "s1_b": 300.0, "s2_a": 10.0}[r]}
        if with_demo:
            row.update({"gender": "Male" if s == "s1" else "Female", "age": 30.0 if s == "s1" else 45.0,
                        "height": 170.0, "weight": 70.0})
        meta.append(row)
    X = S.zscore_rows(rng.standard_normal((6, 320)))
    t = np.array([0, 0, 300, 300, 10, 10], np.float64) if with_time else None
    return WindowSet(X=X, y=y, subject=subj, record=rec, time=t, meta=pd.DataFrame(meta), fs=64, source="toy")


class TestWindowSet:
    def test_basic_properties_and_describe(self):
        ws = _toy_ws()
        assert len(ws) == 6 and ws.n_records == 3 and ws.n_subjects == 2 and ws.window_len == 320
        assert ws.has_time and ws.fs == 64
        assert ws.X.dtype == np.float32 and ws.y.dtype == np.float32
        assert ws.describe() == legacy.WindowSet(ws.X, ws.y, ws.subject, ws.record, ws.meta, "toy").describe()
        assert "6 windows, 3 records, 2 subjects" in ws.describe()
        assert ws.to_dict()["n_records"] == 3

    def test_validation(self):
        ws = _toy_ws()
        with pytest.raises(ValueError):
            WindowSet(ws.X, ws.y[:-1], ws.subject, ws.record)
        with pytest.raises(ValueError):
            WindowSet(ws.X, ws.y, ws.subject, ws.record, time=np.zeros(5))
        with pytest.raises(ValueError, match="meta row"):
            WindowSet(ws.X, ws.y, ws.subject, ws.record, meta=ws.meta.iloc[:1])
        bad = ws.X.copy(); bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            WindowSet(bad, ws.y, ws.subject, ws.record)
        dup = pd.concat([ws.meta, ws.meta.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="one row per record"):
            WindowSet(ws.X, ws.y, ws.subject, ws.record, meta=dup)

    def test_subset_mask_and_index(self):
        ws = _toy_ws()
        sub = ws.subset(ws.subject == "s1")
        assert len(sub) == 4 and sub.n_records == 2 and set(sub.meta["record"]) == {"s1_a", "s1_b"}
        np.testing.assert_array_equal(sub.time, [0, 0, 300, 300])
        sub2 = ws.subset(np.array([5, 0]))                         # order preserved
        assert sub2.record.tolist() == ["s2_a", "s1_a"] and sub2.y.tolist() == [90, 100]
        assert ws[2:4].record.tolist() == ["s1_b", "s1_b"]
        assert ws.subset(np.zeros(6, bool)).n_records == 0 and len(ws.subset(np.zeros(6, bool)).meta) == 0
        with pytest.raises(ValueError):
            ws.subset(np.ones(5, bool))
        assert _toy_ws(with_time=False).subset([0, 1]).time is None

    def test_record_and_subject_rows(self):
        ws = _toy_ws()
        rr = ws.record_rows()
        assert list(rr) == ["s1_a", "s1_b", "s2_a"]
        np.testing.assert_array_equal(rr["s1_b"], [2, 3])
        sr = ws.subject_rows()
        np.testing.assert_array_equal(sr["s2"], [4, 5])

    def test_demographics_reproduces_run_subject_cv(self):
        ws = _toy_ws()
        demo = ws.demographics()                                   # default sex_col="sex" falls back to "gender"
        m = ws.meta.set_index("record")
        ref = m.loc[ws.record, ["age", "height", "weight"]].to_numpy(np.float32)
        ref = np.hstack([ref, (m.loc[ws.record, "gender"] == "Male").to_numpy()[:, None]])
        assert demo.shape == (6, 4) and demo.dtype == np.float32
        np.testing.assert_array_equal(demo, ref.astype(np.float32))
        assert ws.demographics(sex_col="gender").shape == (6, 4)
        assert _toy_ws(with_demo=False).demographics() is None     # BIG IDEAs-like: no demographics
        assert ws.demographics(cols=("age",)).shape == (6, 2)

    def test_meta_per_window_errors(self):
        ws = _toy_ws()
        assert ws.meta_per_window("glucose")["glucose"].tolist() == ws.y.tolist()
        with pytest.raises(KeyError):
            ws.meta_per_window("not_a_column")

    def test_feature_bundle(self):
        ws = _toy_ws()
        fb = F.feature_bundle(ws)
        assert fb["hand"].shape == (6, 9) and fb["demo"].shape == (6, 4) and "morph" not in fb
        assert F.feature_bundle(_toy_ws(with_demo=False))["demo"] is None

    def test_save_load_roundtrip(self, tmp_path):
        ws = _toy_ws()
        p = ws.save(tmp_path / "toy.npz")
        back = WindowSet.load(p)
        np.testing.assert_array_equal(back.X, ws.X)
        np.testing.assert_array_equal(back.time, ws.time)
        assert back.record.tolist() == ws.record.tolist() and back.source == "toy" and back.fs == 64
        assert set(back.meta.columns) == set(ws.meta.columns) and len(back.meta) == 3

    def test_concat(self):
        a, b = _toy_ws().subset(np.array([0, 1])), _toy_ws().subset(np.array([4, 5]))
        c = concat_windowsets([a, b], source="joined")
        assert len(c) == 4 and c.n_records == 2 and c.source == "joined" and c.time is not None
        assert concat_windowsets([a, _toy_ws(with_time=False).subset([4])]).time is None
        with pytest.raises(ValueError):
            concat_windowsets([])
