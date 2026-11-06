"""
ppgbench.models.pretrained
==========================
Frozen-encoder *probes*: a fixed embedding of each PPG window followed by a
ridge (or small MLP) head fitted on the training fold only.

Scientific rationale
--------------------
Foundation models for PPG (PaPaGei, AnyPPG, Pulse-PPG, ...) are trained on
millions of unlabeled pulses.  The honest way to ask whether they carry
glucose information is a *linear probe* under the same leakage-controlled
protocol as every other model: freeze the encoder, embed every window once,
fit a ridge regression on the training rows, score the held-out subjects at
record level.  If a probe cannot beat the ``mean``/``demo`` nulls, neither
representation learning nor fine-tuning is justified on this data.

Contents
--------
* :class:`Encoder` protocol - ``embed(X, fs) -> (N, d)``.
* :class:`IdentityEncoder` - the 9 hand-crafted features (makes
  ``probe_ridge`` identical to ``ridge``; useful as a test oracle).
* :class:`RawEncoder` - decimated raw samples (a "no feature engineering"
  reference).
* :class:`TorchEncoder` - any ``nn.Module`` exposing ``embed(x1)`` (e.g. a
  trained :class:`~ppgbench.models.nets.FusionNet` branch) run frozen.
* :class:`PaPaGeiEncoder` - adapter stub.  Resolves weights via the
  ``PPGBENCH_PAPAGEI`` environment variable and raises a precise
  ``ImportError``/``NotImplementedError`` until the weights and package are
  available.
* :class:`ProbeModel` - the classical ``fit``/``predict`` interface over any
  encoder, with per-``WindowSet`` embedding cache so LOSO does not recompute
  embeddings 23 times.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ppgbench.models.classical import RIDGE_ALPHAS, ClassicalModel
from ppgbench.models.registry import register_model

log = logging.getLogger("ppgbench.models.pretrained")

PAPAGEI_ENV = "PPGBENCH_PAPAGEI"


class PretrainedWeightsMissing(ImportError):
    """Raised when a pretrained encoder's package or weights cannot be found."""


# --------------------------------------------------------------------------- #
# Encoder protocol and simple encoders
# --------------------------------------------------------------------------- #
@runtime_checkable
class Encoder(Protocol):
    """Anything that maps z-scored windows ``X (N, L)`` to embeddings ``(N, d)``.

    Implementations must be deterministic and must not look at labels.
    """

    name: str

    def embed(self, X: np.ndarray, fs: int = 64) -> np.ndarray: ...


def _handcrafted(X: np.ndarray, fs: int) -> np.ndarray:
    """Hand-crafted features from ``ppgbench.core.features`` (fallback: ``ppg_data``)."""
    try:
        from ppgbench.core.features import handcrafted_features   # type: ignore
        return np.asarray(handcrafted_features(X, fs=fs), np.float32)
    except ImportError:
        import ppg_data   # type: ignore  (root-level legacy module; sys.path handled by nets)
        return np.asarray(ppg_data.handcrafted_features(X), np.float32)


class IdentityEncoder:
    """Hand-crafted 9-feature "encoder" - a probe on it equals the ``ridge`` baseline."""

    name = "identity"
    dim = 9

    def embed(self, X: np.ndarray, fs: int = 64) -> np.ndarray:
        import ppgbench.models.nets  # noqa: F401  ensures the repo root is importable for the fallback
        return _handcrafted(np.asarray(X, np.float32), fs)


class RawEncoder:
    """Decimated raw samples: ``X[:, ::stride]`` (no learned representation at all)."""

    name = "raw"

    def __init__(self, stride: int = 4) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride
        self.name = f"raw{stride}"

    def embed(self, X: np.ndarray, fs: int = 64) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(X, np.float32)[:, ::self.stride])


class TorchEncoder:
    """Run a frozen ``torch.nn.Module`` as an encoder.

    The module must expose ``embed(x1)`` (``x1``: (B, 1, L)) or, failing that,
    ``forward(x1)`` returning ``(B, d)``.  Used to probe a network trained on
    another cohort, or a third-party checkpoint wrapped in a thin module.
    """

    def __init__(self, module: Any, name: str = "torch", batch: int = 256, device: Any = None) -> None:
        import torch
        self.module, self.name, self.batch = module.eval(), name, batch
        self.device = torch.device(device) if device is not None else next(module.parameters()).device

    def embed(self, X: np.ndarray, fs: int = 64) -> np.ndarray:
        import torch
        fn = getattr(self.module, "embed", None) or self.module
        out = []
        with torch.no_grad():
            for b in range(0, len(X), self.batch):
                x1 = torch.as_tensor(np.asarray(X[b:b + self.batch], np.float32))[:, None, :].to(self.device)
                out.append(fn(x1).float().cpu())
        return torch.cat(out).numpy() if out else np.empty((0, 0), np.float32)


class PaPaGeiEncoder:
    """Adapter for the PaPaGei PPG foundation model (Pillai et al., 2024).

    Weights are looked up at ``PPGBENCH_PAPAGEI`` (a directory or ``.pt`` file).
    The adapter is a *stub*: constructing it checks that the ``papagei``
    package and the weights exist and otherwise raises
    :class:`PretrainedWeightsMissing` (an ``ImportError``) with instructions;
    :meth:`embed` raises ``NotImplementedError`` until the forward pass is
    wired to the released model code.
    """

    name = "papagei"
    dim = 512
    native_fs = 125

    def __init__(self, weights: str | os.PathLike | None = None, device: Any = None) -> None:
        path = str(weights) if weights is not None else os.environ.get(PAPAGEI_ENV, "")
        if not path:
            raise PretrainedWeightsMissing(
                f"PaPaGei weights not configured: set ${PAPAGEI_ENV} to the downloaded checkpoint "
                "(https://github.com/Nokia-Bell-Labs/papagei-foundation-model) or pass weights=...")
        if not os.path.exists(path):
            raise PretrainedWeightsMissing(f"PaPaGei weights not found at {path!r} (from ${PAPAGEI_ENV})")
        try:
            import papagei  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise PretrainedWeightsMissing(
                "the 'papagei' package is not installed in this environment; install it from the "
                "official repository before using PaPaGeiEncoder") from exc
        self.weights, self.device = path, device

    def embed(self, X: np.ndarray, fs: int = 64) -> np.ndarray:
        raise NotImplementedError(
            "PaPaGeiEncoder.embed is a stub: resample windows to 125 Hz, run papagei's ResNet1D-MoE "
            "encoder with the checkpoint at %r and return (N, 512) embeddings" % self.weights)


ENCODERS: dict[str, Any] = {"identity": IdentityEncoder, "raw": RawEncoder, "papagei": PaPaGeiEncoder}


def get_encoder(encoder: str | Encoder, **kw: Any) -> Encoder:
    """Resolve ``encoder`` (name or instance) to an :class:`Encoder`."""
    if isinstance(encoder, str):
        if encoder not in ENCODERS:
            raise KeyError(f"unknown encoder {encoder!r}; known: {sorted(ENCODERS)}")
        return ENCODERS[encoder](**kw)
    if not hasattr(encoder, "embed"):
        raise TypeError(f"{encoder!r} does not implement Encoder.embed")
    return encoder


# --------------------------------------------------------------------------- #
# Probe model
# --------------------------------------------------------------------------- #
@register_model("probe_ridge", "probe", aliases=("probe",), doc="Frozen encoder + StandardScaler + RidgeCV head")
class ProbeModel(ClassicalModel):
    """Frozen encoder + ridge or small-MLP head with the classical interface.

    Parameters
    ----------
    encoder:
        Encoder name (``"identity"``, ``"raw"``, ``"papagei"``) or instance.
    head:
        ``"ridge"`` (StandardScaler + ``RidgeCV``, alpha grid as the ``ridge``
        baseline) or ``"mlp"`` (StandardScaler + ``MLPRegressor`` on the
        standardised target; one hidden layer, early stopping off so the
        budget is fixed and no held-out data is touched).
    alphas, hidden, max_iter, seed:
        Head hyper-parameters.
    demo:
        If True, concatenate ``feats["demo"]`` to the embedding.
    encoder_kw:
        Forwarded to the encoder constructor when ``encoder`` is a name.

    Embeddings are cached per ``WindowSet`` (keyed on the identity and shape of
    ``ws.X``, and holding a reference to that array so a recycled ``id`` can
    never alias another set) so that the same probe object can be re-used
    across folds; the fit itself only ever sees the training rows.
    """

    inputs: tuple[str, ...] = ()

    def __init__(self, encoder: str | Encoder = "identity", head: str = "ridge",
                 alphas: np.ndarray | None = None, hidden: int = 64, max_iter: int = 500, seed: int = 0,
                 demo: bool = False, **encoder_kw: Any) -> None:
        super().__init__()
        if head not in ("ridge", "mlp"):
            raise ValueError("head must be 'ridge' or 'mlp'")
        self.encoder = get_encoder(encoder, **encoder_kw)
        self.head, self.hidden, self.max_iter, self.seed, self.demo = head, hidden, max_iter, seed, demo
        self.alphas = RIDGE_ALPHAS if alphas is None else np.asarray(alphas, np.float64)
        self.inputs = ("demo",) if demo else ()
        self._cache: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}

    # -- embedding -----------------------------------------------------------
    def embeddings(self, ws: Any) -> np.ndarray:
        """(N, d) embedding of every window in ``ws`` (cached)."""
        X = np.asarray(ws.X)
        key = (id(X), X.shape)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is X:               # same array object, not merely a recycled id
            return hit[1]
        Z = np.asarray(self.encoder.embed(X, int(getattr(ws, "fs", 64))), np.float64)
        if Z.ndim != 2 or Z.shape[0] != len(X):
            raise ValueError(f"encoder {self.encoder.name!r} returned shape {Z.shape}, expected ({len(X)}, d)")
        Z = np.nan_to_num(Z)
        self._cache.clear()
        self._cache[key] = (X, Z)                          # keeping X alive pins its id for the cache's lifetime
        log.debug("embedded %d windows with %s -> d=%d", len(X), self.encoder.name, Z.shape[1])
        return Z

    def _design(self, ws: Any, feats: Mapping[str, Any]) -> np.ndarray:
        Z = self.embeddings(ws)
        if self.demo:
            from ppgbench.models.classical import assemble_features
            Z = np.hstack([Z, assemble_features(feats, ("demo",))])
        return Z

    # -- fit / predict -------------------------------------------------------
    def _fit(self, ws: Any, tr: np.ndarray, feats: Mapping[str, Any]) -> None:
        Z = self._design(ws, feats)
        y = np.asarray(ws.y, np.float64)[tr]
        self.scaler_ = StandardScaler().fit(Z[tr])
        Ztr = self.scaler_.transform(Z[tr])
        if self.head == "ridge":
            self.model_ = RidgeCV(alphas=self.alphas).fit(Ztr, y)
            self.y_mu_, self.y_sd_ = 0.0, 1.0
        else:
            self.y_mu_, self.y_sd_ = float(y.mean()), float(y.std() + 1e-6)
            self.model_ = MLPRegressor(hidden_layer_sizes=(self.hidden,), max_iter=self.max_iter,
                                       random_state=self.seed, early_stopping=False, alpha=1e-3)
            self.model_.fit(Ztr, (y - self.y_mu_) / self.y_sd_)

    def _predict(self, ws: Any, te: np.ndarray, feats: Mapping[str, Any]) -> np.ndarray:
        Z = self._design(ws, feats)
        p = self.model_.predict(self.scaler_.transform(Z[te]))
        return (p * self.y_sd_ + self.y_mu_).astype(np.float32)

    def __repr__(self) -> str:
        return f"ProbeModel(encoder={self.encoder.name!r}, head={self.head!r}, demo={self.demo}, fitted={self._fitted})"


register_model("probe_mlp", "probe", doc="Frozen encoder + StandardScaler + 1-hidden-layer MLP head")(
    lambda encoder="identity", **kw: ProbeModel(encoder, head="mlp", **kw))
register_model("probe_raw", "probe", doc="Decimated raw samples + RidgeCV (no feature engineering)")(
    lambda **kw: ProbeModel("raw", head="ridge", **kw))
register_model("papagei_probe", "probe", doc="PaPaGei foundation-model embeddings + RidgeCV (needs weights)")(
    lambda head="ridge", **kw: ProbeModel("papagei", head=head, **kw))


__all__ = ["Encoder", "IdentityEncoder", "RawEncoder", "TorchEncoder", "PaPaGeiEncoder", "ProbeModel",
           "PretrainedWeightsMissing", "ENCODERS", "get_encoder", "PAPAGEI_ENV"]
