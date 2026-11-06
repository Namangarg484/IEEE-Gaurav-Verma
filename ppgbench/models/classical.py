"""
ppgbench.models.classical
=========================
Null and classical baselines that every deep result must be compared against.

Scientific rationale
--------------------
A PPG->glucose regressor is only interesting if it beats (i) the *null* that
predicts the training mean (what "no signal" looks like at the record level),
(ii) the *demographic* model (age/sex/height/weight explain a good share of
between-subject glucose variance without touching the waveform), and
(iii) a strong *classical* learner on hand-crafted PPG descriptors (heart rate,
spectral bands, morphology moments).  The leaky ``subject_mean`` predictor is
the memorisation ceiling a record-level leak allows: a model that merely
recognises *who* the window belongs to can achieve it, so any protocol that
lets ``subject_mean`` beat ``mean`` leaks subject identity.

All models share one interface (binding, see ``docs/framework_spec.md``)::

    model.fit(ws, tr_idx, feats) -> model
    model.predict(ws, te_idx, feats) -> np.ndarray (len(te_idx),) float32

where ``feats = {"hand": (N, 9), "demo": (N, k) | None, "morph": optional}``.

Numerical parity
----------------
``ridge``, ``demo``, ``ridge+demo``, ``gbm``, ``gbm+demo`` and ``mean``
reproduce ``run_subject_cv.predict_classical`` exactly (same scaler, same
``RidgeCV`` alpha grid, same ``HistGradientBoostingRegressor``
hyper-parameters and ``random_state``, and the same float32 feature matrix,
so the out-of-fold predictions are bit-identical; verified on the real
Mendeley data by ``tests/ppgbench/test_parity.py``).  ``subject_mean`` reproduces
``run_leakage_ladder.leaky_subject_mean`` (fallback to the global train mean)
and, with ``fallback="error"``, ``run_bigideas_cv.subject_mean_baseline``.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ppgbench.models.registry import register_model

log = logging.getLogger("ppgbench.models.classical")

Feats = Mapping[str, Any]

#: ``RidgeCV`` alpha grid used by every ridge variant (legacy value).
RIDGE_ALPHAS: np.ndarray = np.logspace(-2, 3, 12)

#: ``HistGradientBoostingRegressor`` settings used by every gbm variant (legacy values).
GBM_PARAMS: dict[str, Any] = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                  min_samples_leaf=10, l2_regularization=1.0, random_state=0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_index(idx: Any, n: int) -> np.ndarray:
    """Normalise a boolean mask or integer index array to ``int64`` positions."""
    idx = np.asarray(idx)
    if idx.dtype == bool:
        if idx.shape != (n,):
            raise ValueError(f"boolean index has shape {idx.shape}, expected ({n},)")
        return np.flatnonzero(idx)
    return idx.astype(np.int64, copy=False)


def assemble_features(feats: Feats, inputs: Sequence[str]) -> np.ndarray:
    """Horizontally stack ``feats[k]`` for ``k in inputs`` into one (N, d) float matrix.

    The floating dtype of the blocks is preserved (numpy promotion applies when
    they differ).  The hand-crafted and demographic blocks are float32, so
    ``StandardScaler`` / ``RidgeCV`` run in float32 exactly as
    ``run_subject_cv.predict_classical`` did and the legacy out-of-fold
    predictions are reproduced bit for bit (``tests/ppgbench/test_parity.py``);
    up-casting to float64 here would move every ridge prediction by about one
    float32 ulp (~1e-5 mg/dL).  Boolean / integer-only matrices are cast to
    float64.  A 1-D block is treated as a single column.

    Raises ``ValueError`` with an explicit message when a requested block is
    missing or ``None`` (e.g. a ``demo`` model on a dataset without
    demographics), instead of letting ``np.hstack`` fail obscurely.
    """
    blocks = []
    for key in inputs:
        block = feats.get(key) if isinstance(feats, Mapping) else None
        if block is None:
            raise ValueError(f"feature block {key!r} is required but missing/None in feats "
                             f"(available: {sorted(k for k, v in feats.items() if v is not None)})")
        block = np.asarray(block)
        if block.ndim == 1:
            block = block[:, None]
        if block.ndim != 2:
            raise ValueError(f"feature block {key!r} must be 1-D or 2-D, got shape {block.shape}")
        blocks.append(block)
    F = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    if F.dtype.kind != "f":                        # bool / int only: no float block to promote them
        F = F.astype(np.float64)
    if not np.isfinite(F).all():
        n_bad = int((~np.isfinite(F)).any(1).sum())
        log.warning("%d rows contain non-finite features; replacing with 0", n_bad)
        F = np.nan_to_num(F)
    return F


class ClassicalModel:
    """Base class: common ``fit``/``predict``/``fit_predict`` plumbing.

    Subclasses implement :meth:`_fit` and :meth:`_predict`.  Fitted state is
    held on the instance, so build a fresh model per fold (``build_model``
    always does).
    """

    #: Feature blocks the model consumes (empty for label-only nulls).
    inputs: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._fitted = False

    # -- interface ----------------------------------------------------------
    def fit(self, ws: Any, tr_idx: Any, feats: Feats | None = None) -> "ClassicalModel":
        """Fit on the training rows ``tr_idx`` of ``ws`` (using ``feats`` when needed)."""
        tr = _as_index(tr_idx, len(ws))
        if len(tr) == 0:
            raise ValueError("empty training index")
        self._fit(ws, tr, feats or {})
        self._fitted = True
        return self

    def predict(self, ws: Any, te_idx: Any, feats: Feats | None = None) -> np.ndarray:
        """Predict glucose (mg/dL) for rows ``te_idx`` of ``ws``; returns float32 (len(te),)."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predict")
        te = _as_index(te_idx, len(ws))
        if len(te) == 0:
            return np.empty(0, np.float32)
        return np.asarray(self._predict(ws, te, feats or {}), dtype=np.float32).reshape(-1)

    def fit_predict(self, ws: Any, tr_idx: Any, te_idx: Any, feats: Feats | None = None) -> np.ndarray:
        """``fit`` then ``predict`` in one call (the per-fold pattern)."""
        return self.fit(ws, tr_idx, feats).predict(ws, te_idx, feats)

    # -- to implement --------------------------------------------------------
    def _fit(self, ws: Any, tr: np.ndarray, feats: Feats) -> None:
        raise NotImplementedError

    def _predict(self, ws: Any, te: np.ndarray, feats: Feats) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(inputs={self.inputs}, fitted={self._fitted})"


# --------------------------------------------------------------------------- #
# Null models
# --------------------------------------------------------------------------- #
@register_model("mean", "classical", doc="Null: global mean of the training labels")
class MeanRegressor(ClassicalModel):
    """Predict the mean training glucose for every test window.

    This is the correct null for subject-disjoint protocols: its record-level
    MAE is the dispersion of glucose in the population and any model must beat
    it with a confidence interval that excludes zero.
    """

    def _fit(self, ws: Any, tr: np.ndarray, feats: Feats) -> None:
        self.mean_ = float(np.asarray(ws.y, np.float64)[tr].mean())

    def _predict(self, ws: Any, te: np.ndarray, feats: Feats) -> np.ndarray:
        return np.full(len(te), self.mean_, np.float32)


@register_model("subject_mean", "classical", aliases=("subjmean", "subject-mean"),
                doc="Leaky null: the test window's OWN subject's training mean (memorisation ceiling)")
class SubjectMeanRegressor(ClassicalModel):
    """Per-subject training mean, i.e. what a model that recognises identity attains.

    Parameters
    ----------
    fallback:
        ``"global"`` (default): subjects absent from training receive the
        global training mean - matches ``run_leakage_ladder.leaky_subject_mean``
        and keeps the predictor well defined under subject-disjoint splits
        (where it collapses to :class:`MeanRegressor`).
        ``"error"``: raise ``ValueError`` for such subjects - matches the
        strictness of ``run_bigideas_cv.subject_mean_baseline`` used as the
        within-subject null on chronological splits.
    """

    def __init__(self, fallback: str = "global") -> None:
        super().__init__()
        if fallback not in ("global", "error"):
            raise ValueError("fallback must be 'global' or 'error'")
        self.fallback = fallback

    def _fit(self, ws: Any, tr: np.ndarray, feats: Feats) -> None:
        y = np.asarray(ws.y, np.float64)[tr]
        s = np.asarray(ws.subject)[tr]
        self.global_mean_ = float(y.mean())
        self.subject_means_ = {str(k): float(y[s == k].mean()) for k in np.unique(s)}

    def _predict(self, ws: Any, te: np.ndarray, feats: Feats) -> np.ndarray:
        s = np.asarray(ws.subject)[te]
        pred = np.full(len(te), self.global_mean_, np.float32)
        missing = []
        for k in np.unique(s):
            m = self.subject_means_.get(str(k))
            if m is None:
                missing.append(str(k))
                continue
            pred[s == k] = m
        if missing:
            if self.fallback == "error":
                raise ValueError(f"subject_mean: test subjects {missing} have no training windows")
            log.debug("subject_mean: %d test subjects unseen in training -> global mean", len(missing))
        return pred


# --------------------------------------------------------------------------- #
# Feature-based learners
# --------------------------------------------------------------------------- #
class RidgeFeatures(ClassicalModel):
    """StandardScaler + ``RidgeCV`` (leave-one-out GCV over :data:`RIDGE_ALPHAS`).

    Ridge on standardised features is deliberately simple: it is linear, has one
    hyper-parameter chosen internally on the *training* rows only, and cannot
    memorise subject identity from 9-13 scalar descriptors.
    """

    def __init__(self, inputs: Sequence[str] = ("hand",), alphas: np.ndarray | None = None) -> None:
        super().__init__()
        self.inputs = tuple(inputs)
        self.alphas = RIDGE_ALPHAS if alphas is None else np.asarray(alphas, np.float64)

    def _fit(self, ws: Any, tr: np.ndarray, feats: Feats) -> None:
        F = assemble_features(feats, self.inputs)
        self.scaler_ = StandardScaler().fit(F[tr])
        self.model_ = RidgeCV(alphas=self.alphas).fit(self.scaler_.transform(F[tr]), np.asarray(ws.y)[tr])

    def _predict(self, ws: Any, te: np.ndarray, feats: Feats) -> np.ndarray:
        F = assemble_features(feats, self.inputs)
        return self.model_.predict(self.scaler_.transform(F[te])).astype(np.float32)

    @property
    def alpha_(self) -> float:
        """Selected ridge penalty (after ``fit``)."""
        return float(self.model_.alpha_)


class GBMFeatures(ClassicalModel):
    """``HistGradientBoostingRegressor`` with the legacy, deliberately regularised settings.

    Trees are the standard "strong classical" reference for tabular PPG
    features; no scaling is needed.  ``random_state=0`` makes the fit
    deterministic regardless of the fold seed, exactly as the legacy script.
    """

    def __init__(self, inputs: Sequence[str] = ("hand",), **params: Any) -> None:
        super().__init__()
        self.inputs = tuple(inputs)
        self.params = {**GBM_PARAMS, **params}

    def _fit(self, ws: Any, tr: np.ndarray, feats: Feats) -> None:
        F = assemble_features(feats, self.inputs)
        self.model_ = HistGradientBoostingRegressor(**self.params).fit(F[tr], np.asarray(ws.y)[tr])

    def _predict(self, ws: Any, te: np.ndarray, feats: Feats) -> np.ndarray:
        F = assemble_features(feats, self.inputs)
        return self.model_.predict(F[te]).astype(np.float32)


# Registered variants mirror run_subject_cv.CLASSICAL exactly.
register_model("ridge", "classical", doc="RidgeCV on 9 hand-crafted PPG features",
               views=("hand",))(lambda **kw: RidgeFeatures(("hand",), **kw))
register_model("demo", "classical", doc="RidgeCV on demographics only (age, height, weight, sex)",
               views=("demo",))(lambda **kw: RidgeFeatures(("demo",), **kw))
register_model("ridge+demo", "classical", aliases=("ridge_demo",),
               doc="RidgeCV on hand-crafted features + demographics",
               views=("hand", "demo"))(lambda **kw: RidgeFeatures(("hand", "demo"), **kw))
register_model("gbm", "classical", doc="Gradient-boosted trees on 9 hand-crafted PPG features",
               views=("hand",))(lambda **kw: GBMFeatures(("hand",), **kw))
register_model("gbm+demo", "classical", aliases=("gbm_demo",),
               doc="Gradient-boosted trees on hand-crafted features + demographics",
               views=("hand", "demo"))(lambda **kw: GBMFeatures(("hand", "demo"), **kw))
register_model("ridge+morph", "classical", aliases=("ridge_morph",),
               doc="RidgeCV on hand-crafted + neurokit morphology features",
               views=("hand", "morph"))(lambda **kw: RidgeFeatures(("hand", "morph"), **kw))
register_model("gbm+morph", "classical", aliases=("gbm_morph",),
               doc="Gradient-boosted trees on hand-crafted + neurokit morphology features",
               views=("hand", "morph"))(lambda **kw: GBMFeatures(("hand", "morph"), **kw))


# --------------------------------------------------------------------------- #
# One-call convenience (legacy signature)
# --------------------------------------------------------------------------- #
def predict_classical(name: str, ws: Any, tr: np.ndarray, te: np.ndarray, feats: Feats, **kw: Any) -> np.ndarray:
    """Build, fit and predict a classical model in one call.

    Equivalent to ``run_subject_cv.predict_classical(name, feats["hand"],
    feats["demo"], ws.y, tr, te)`` but routed through the registry so that
    ``subject_mean`` and future models work the same way.
    """
    from ppgbench.models.registry import build_model, model_kind
    if model_kind(name) not in ("classical", "probe"):
        raise ValueError(f"{name!r} is a {model_kind(name)} model; use ppgbench.train.train_net")
    return build_model(name, **kw).fit(ws, tr, feats).predict(ws, te, feats)


__all__ = ["ClassicalModel", "MeanRegressor", "SubjectMeanRegressor", "RidgeFeatures", "GBMFeatures",
           "assemble_features", "predict_classical", "RIDGE_ALPHAS", "GBM_PARAMS"]
