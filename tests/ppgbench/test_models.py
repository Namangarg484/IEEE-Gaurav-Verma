"""
Tests for ppgbench.models (registry, classical, nets, pretrained) and ppgbench.train.

Everything runs on CPU with tiny synthetic tensors and 1 epoch; the whole
module takes a few seconds.  Numerical-parity tests against the legacy
scripts (``run_subject_cv.train_fold`` / ``predict_classical``) use the same
tiny data so they are fast and run unconditionally.

Run:  ~/.venvs/ppgfusion/bin/python -m pytest tests/ppgbench/test_models.py -q
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)          # tiny tensors: avoid OpenMP oversubscription when suites run concurrently

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppgbench.models import (  # noqa: E402
    ENCODERS, FusionNet, IdentityEncoder, MeanRegressor, PaPaGeiEncoder, PretrainedWeightsMissing, ProbeModel,
    RawEncoder, SingleBranch, SubjectMeanRegressor, TinyTransformer, TorchEncoder, build_model, count_params,
    get_spec, list_models, model_kind, predict_classical, register_model, required_views, resolve_name,
)
from ppgbench.train import TrainConfig, device, predict_net, seed_everything, train_fold, train_net  # noqa: E402

L, N_SUBJ, REC_PER_SUBJ, WIN_PER_REC = 320, 4, 3, 4
CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@dataclass
class _WS:
    """Minimal stand-in with the WindowSet fields the models read."""

    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    record: np.ndarray
    time: np.ndarray | None = None
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    fs: int = 64
    source: str = "synthetic-test"

    def __len__(self) -> int:
        return len(self.y)


def _make_ws(seed: int = 0) -> _WS:
    """Try the package's synthetic dataset; fall back to a local generator."""
    try:
        from ppgbench.data import load_dataset  # type: ignore
        ws = load_dataset("synthetic", n_subjects=N_SUBJ, records_per_subject=REC_PER_SUBJ,
                          windows_per_record=WIN_PER_REC, seed=seed)
        if ws.X.shape[1] == L and len(ws) >= 32:
            return ws
    except Exception:
        pass
    rng = np.random.RandomState(seed)
    n = N_SUBJ * REC_PER_SUBJ * WIN_PER_REC
    t = np.arange(L) / 64.0
    X = np.empty((n, L), np.float32)
    y = np.empty(n, np.float32)
    subject, record = np.empty(n, object), np.empty(n, object)
    k = 0
    for s in range(N_SUBJ):
        hr = 1.0 + 0.3 * s
        off = 90.0 + 15.0 * s
        for r in range(REC_PER_SUBJ):
            g = off + rng.randn() * 8
            for _ in range(WIN_PER_REC):
                x = np.sin(2 * np.pi * hr * t + rng.rand() * 6.28) + 0.3 * rng.randn(L)
                X[k] = (x - x.mean()) / (x.std() + 1e-8)
                y[k], subject[k], record[k] = g + 0.5 * X[k, :10].sum(), f"S{s:02d}", f"S{s:02d}_R{r:02d}"
                k += 1
    meta = pd.DataFrame({"record": np.unique(record)})
    meta["subject"] = meta.record.str.slice(0, 3)
    meta["glucose"] = [float(y[record == r][0]) for r in meta.record]
    return _WS(X, y, subject.astype(str), record.astype(str), None, meta)


@pytest.fixture(scope="module")
def ws():
    return _make_ws()


@pytest.fixture(scope="module")
def views(ws):
    g = torch.Generator().manual_seed(0)
    n = len(ws)
    return {"x1": torch.from_numpy(ws.X[:, None, :].astype(np.float32)),
            "x2": torch.rand(n, 1, 64, 64, generator=g),
            "x3": torch.randn(n, 3, 256, generator=g)}


@pytest.fixture(scope="module")
def feats(ws):
    rng = np.random.RandomState(1)
    hand = IdentityEncoder().embed(ws.X, ws.fs)
    demo = rng.rand(len(ws), 4).astype(np.float32)
    return {"hand": hand, "demo": demo, "morph": None}


@pytest.fixture(scope="module")
def fold(ws):
    """Train on subjects S00..S02, test on S03 (subject-disjoint)."""
    te = np.flatnonzero(ws.subject == "S03")
    tr = np.flatnonzero(ws.subject != "S03")
    return tr, te


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_lists_all_required_models():
    models = list_models()
    for n in ("mean", "subject_mean", "ridge", "demo", "ridge+demo", "gbm", "gbm+demo"):
        assert models[n] == "classical", n
    for n in ("fusion", "1d", "2d", "3d", "resnet1d", "cnnlstm", "tiny_transformer"):
        assert models[n] == "net", n
    for n in ("probe_ridge", "probe_mlp", "papagei_probe"):
        assert models[n] == "probe", n
    assert list_models(kind="net") == {k: v for k, v in models.items() if v == "net"}


def test_registry_aliases_and_errors():
    assert resolve_name("subjmean") == "subject_mean"
    assert resolve_name("transformer") == "tiny_transformer"
    assert model_kind("ridge_demo") == "classical"
    with pytest.raises(KeyError, match="unknown model"):
        build_model("fusionn")
    with pytest.raises(ValueError):
        register_model("mean", "classical")(lambda: None)      # different factory -> refuse override
    with pytest.raises(ValueError):
        register_model("x_bad_kind", "weird")(lambda: None)


def test_build_model_returns_fresh_instances():
    a, b = build_model("fusion"), build_model("fusion")
    assert a is not b and isinstance(a, FusionNet)
    assert isinstance(build_model("1d"), SingleBranch) and build_model("3d").which == "3d"
    assert isinstance(build_model("mean"), MeanRegressor)


# --------------------------------------------------------------------------- #
# Nets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["fusion", "1d", "2d", "3d", "resnet1d", "cnnlstm", "tiny_transformer"])
def test_net_forward_shape(name, views):
    torch.manual_seed(0)
    m = build_model(name).eval()
    with torch.no_grad():
        out = m(views["x1"][:4], views["x2"][:4], views["x3"][:4])
    assert out.shape == (4, 1) and torch.isfinite(out).all()
    assert count_params(m) > 0
    assert set(required_views(name)) <= {"x1", "x2", "x3"}


def test_single_branch_ignores_unused_views(views):
    m = build_model("1d").eval()
    with torch.no_grad():
        a = m(views["x1"][:4], None, None)
        b = m(views["x1"][:4], views["x2"][:4], views["x3"][:4])
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        m(None, views["x2"][:4], None)


def test_fusion_wrapper_matches_legacy_init_and_exposes_embed(views):
    import fusion_network_model as legacy
    torch.manual_seed(3); ours = FusionNet()
    torch.manual_seed(3); theirs = legacy.MultimodalFusionNet()
    assert [k for k, _ in ours.state_dict().items()] == [k for k, _ in theirs.state_dict().items()]
    for (k, a), (_, b) in zip(ours.state_dict().items(), theirs.state_dict().items()):
        assert torch.equal(a, b), k
    ours.eval()
    with torch.no_grad():
        z = ours.embed(views["x1"][:4], views["x2"][:4], views["x3"][:4])
        w = ours.branch_weights(views["x1"][:4], views["x2"][:4], views["x3"][:4])
    assert z.shape == (4, 768) and w.shape == (4, 3)
    assert torch.allclose(w.sum(1), torch.ones(4), atol=1e-5)


def test_tiny_transformer_handles_odd_lengths_and_kwargs():
    m = TinyTransformer(patch=16, d_model=32, n_heads=4, n_layers=1).eval()
    with torch.no_grad():
        assert m(torch.randn(3, 1, 320)).shape == (3, 1)
        assert m(torch.randn(3, 1, 333)).shape == (3, 1)          # padded to a multiple of 16
        assert m(torch.randn(3, 320)).shape == (3, 1)             # 2-D input accepted
        assert m.embed(torch.randn(2, 1, 320)).shape == (2, 32)
    with pytest.raises(ValueError):
        m(torch.randn(1, 1, 5000))
    m2 = build_model("tiny_transformer", n_layers=3)
    assert len(m2.encoder.layers) == 3
    with pytest.raises(ValueError):
        TinyTransformer(d_model=30, n_heads=4)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["1d", "tiny_transformer", "fusion"])
def test_train_net_one_epoch_cpu(name, views, ws, fold):
    tr, te = fold
    pred = train_net(name, views, ws.y, tr, te, epochs=1, batch=8, lr=1e-3, seed=0, device=CPU)
    assert pred.shape == (len(te),) and pred.dtype == np.float32 and np.isfinite(pred).all()
    assert 0 < pred.mean() < 300                                  # de-standardised to mg/dL


def test_train_net_is_deterministic_and_seed_sensitive(views, ws, fold):
    tr, te = fold
    a = train_net("1d", views, ws.y, tr, te, epochs=1, batch=8, seed=0, device=CPU)
    b = train_net("1d", views, ws.y, tr, te, epochs=1, batch=8, seed=0, device=CPU)
    c = train_net("1d", views, ws.y, tr, te, epochs=1, batch=8, seed=1, device=CPU)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_train_net_reproduces_legacy_train_fold(views, ws, fold):
    """Bit-for-bit parity with run_subject_cv.train_fold for the same seed/device."""
    import run_subject_cv as legacy
    tr, te = fold
    for name in ("1d", "fusion", "resnet1d"):
        ours = train_net(name, views, ws.y, tr, te, epochs=1, batch=8, lr=1e-3, seed=0, device=CPU)
        theirs = legacy.train_fold(name, views, ws.y, tr, te, 1, 8, 1e-3, 0, CPU)
        assert np.array_equal(ours, theirs), name


def test_train_fold_result_and_prediction_halves(views, ws, fold):
    tr, te = fold
    cfg = TrainConfig(epochs=1, batch=8)
    res = train_fold("1d", views, ws.y, tr, te, cfg, seed=0, dev=CPU)
    assert res.pred.shape == (len(te),) and res.n_steps == len(tr) // 8 and len(res.history) == 1
    assert res.summary()["n_steps"] == res.n_steps
    assert np.isclose(res.y_mean, ws.y[tr].mean()) and np.isclose(res.y_std, ws.y[tr].std() + 1e-6)
    again = predict_net(res.model, views, te, res.y_mean, res.y_std, CPU)
    assert np.array_equal(again, res.pred)
    assert predict_net(res.model, views, te[:0], res.y_mean, res.y_std, CPU).shape == (0,)


def test_train_net_validates_inputs(views, ws, fold):
    tr, te = fold
    with pytest.raises(ValueError, match="kind='net'"):
        train_net("ridge", views, ws.y, tr, te, epochs=1, device=CPU)
    with pytest.raises(ValueError, match="needs view"):
        train_net("fusion", {"x1": views["x1"]}, ws.y, tr, te, epochs=1, device=CPU)
    with pytest.raises(ValueError, match="empty fold"):
        train_net("1d", views, ws.y, tr, te[:0], epochs=1, device=CPU)
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    # 1-D nets train without x2/x3 present at all
    pred = train_net("1d", {"x1": views["x1"]}, ws.y, tr, te, epochs=1, batch=8, device=CPU)
    assert np.isfinite(pred).all()


def test_device_helper_respects_override(monkeypatch):
    assert device("cpu").type == "cpu"
    monkeypatch.setenv("PPGBENCH_DEVICE", "cpu")
    assert device().type == "cpu"
    monkeypatch.delenv("PPGBENCH_DEVICE")
    assert device().type in ("mps", "cpu", "cuda")
    seed_everything(0); a = torch.rand(3)
    seed_everything(0); b = torch.rand(3)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Classical
# --------------------------------------------------------------------------- #
def test_mean_and_subject_mean(ws, fold):
    tr, te = fold
    m = build_model("mean").fit(ws, tr, {})
    p = m.predict(ws, te, {})
    assert p.dtype == np.float32 and np.allclose(p, ws.y[tr].mean())
    # subject-disjoint fold: leaky subject mean collapses to the global mean (fallback)
    sm = build_model("subject_mean").fit(ws, tr, {})
    assert np.allclose(sm.predict(ws, te, {}), ws.y[tr].mean())
    with pytest.raises(ValueError, match="no training windows"):
        SubjectMeanRegressor(fallback="error").fit(ws, tr, {}).predict(ws, te, {})
    # record-level (leaky) fold: each test window gets its own subject's train mean
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ws)); tr2, te2 = perm[: len(ws) // 2], perm[len(ws) // 2:]
    p2 = SubjectMeanRegressor().fit(ws, tr2, {}).predict(ws, te2, {})
    for s in np.unique(ws.subject[te2]):
        assert np.allclose(p2[ws.subject[te2] == s], ws.y[tr2][ws.subject[tr2] == s].mean())
    with pytest.raises(RuntimeError):
        MeanRegressor().predict(ws, te, {})
    mask = np.zeros(len(ws), bool); mask[tr] = True                                # boolean mask accepted
    assert np.allclose(build_model("mean").fit(ws, mask, {}).predict(ws, te, {}), ws.y[tr].mean())


@pytest.mark.parametrize("name", ["ridge", "demo", "ridge+demo", "gbm", "gbm+demo"])
def test_feature_models_match_legacy_predict_classical(name, ws, feats, fold):
    import run_subject_cv as legacy
    tr, te = fold
    ours = build_model(name).fit(ws, tr, feats).predict(ws, te, feats)
    theirs = legacy.predict_classical(name, feats["hand"], feats["demo"], ws.y, tr, te)
    assert ours.dtype == np.float32 and ours.shape == (len(te),)
    assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-4), name
    assert np.allclose(predict_classical(name, ws, tr, te, feats), ours)


def test_feature_models_require_their_blocks(ws, feats, fold):
    tr, te = fold
    with pytest.raises(ValueError, match="'demo' is required"):
        build_model("ridge+demo").fit(ws, tr, {"hand": feats["hand"], "demo": None})
    with pytest.raises(ValueError, match="'morph'"):
        build_model("ridge+morph").fit(ws, tr, feats)
    with pytest.raises(ValueError, match="net model"):
        predict_classical("1d", ws, tr, te, feats)
    r = build_model("ridge").fit(ws, tr, feats)
    assert r.alpha_ in set(np.logspace(-2, 3, 12).round(10)) or r.alpha_ > 0


# --------------------------------------------------------------------------- #
# Probes / pretrained
# --------------------------------------------------------------------------- #
def test_encoders(ws):
    z = IdentityEncoder().embed(ws.X, ws.fs)
    assert z.shape == (len(ws), 9) and np.isfinite(z).all()
    assert RawEncoder(4).embed(ws.X).shape == (len(ws), L // 4)
    net = build_model("1d").eval()
    te = TorchEncoder(net, name="branch1d", device="cpu")
    assert te.embed(ws.X[:5]).shape == (5, 256)
    assert set(ENCODERS) >= {"identity", "raw", "papagei"}
    with pytest.raises(KeyError):
        ProbeModel(encoder="nope")


def test_probe_ridge_equals_ridge_baseline(ws, feats, fold):
    tr, te = fold
    probe = build_model("probe_ridge").fit(ws, tr, feats).predict(ws, te, feats)
    ridge = build_model("ridge").fit(ws, tr, feats).predict(ws, te, feats)
    assert np.allclose(probe, ridge, rtol=1e-5, atol=1e-4)
    assert len(build_model("probe_ridge").fit(ws, tr, feats)._cache) == 1     # embeddings cached once


def test_probe_mlp_and_demo(ws, feats, fold):
    tr, te = fold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                         # 50 iterations: convergence warning expected
        p = build_model("probe_mlp", max_iter=50).fit(ws, tr, feats).predict(ws, te, feats)
    assert p.shape == (len(te),) and np.isfinite(p).all()
    q = ProbeModel("identity", head="ridge", demo=True).fit(ws, tr, feats).predict(ws, te, feats)
    assert np.isfinite(q).all()
    raw = build_model("probe_raw").fit(ws, tr, feats).predict(ws, te, feats)
    assert raw.shape == (len(te),)
    with pytest.raises(ValueError):
        ProbeModel(head="svm")


def test_papagei_adapter_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv("PPGBENCH_PAPAGEI", raising=False)
    with pytest.raises(PretrainedWeightsMissing, match="PPGBENCH_PAPAGEI"):
        PaPaGeiEncoder()
    with pytest.raises(ImportError, match="not found"):
        PaPaGeiEncoder(weights=str(tmp_path / "missing.pt"))
    monkeypatch.setenv("PPGBENCH_PAPAGEI", str(tmp_path / "also_missing.pt"))
    with pytest.raises(ImportError):
        build_model("papagei_probe")
    assert get_spec("papagei_probe").kind == "probe"
