"""
ppgbench.models
===============
Model zoo for the leakage-controlled PPG->glucose benchmark.

Importing this package registers every model::

    >>> from ppgbench.models import build_model, list_models
    >>> list_models()
    {'1d': 'net', '2d': 'net', '3d': 'net', 'cnnlstm': 'net', 'demo': 'classical', ...}

Kinds
-----
``classical``  :mod:`ppgbench.models.classical` - ``mean``, ``subject_mean``,
               ``ridge``, ``demo``, ``ridge+demo``, ``gbm``, ``gbm+demo``
               (``fit(ws, tr, feats)`` / ``predict(ws, te, feats)``).
``net``        :mod:`ppgbench.models.nets` - ``fusion``, ``1d``, ``2d``, ``3d``,
               ``resnet1d``, ``cnnlstm``, ``tiny_transformer``
               (``torch.nn.Module`` with ``forward(x1, x2, x3)``; trained by
               :func:`ppgbench.train.train_net`).
``probe``      :mod:`ppgbench.models.pretrained` - frozen :class:`Encoder` +
               ridge/MLP head (``probe_ridge``, ``probe_mlp``, ``probe_raw``,
               ``papagei_probe``).
"""
from __future__ import annotations

from ppgbench.models.registry import (KINDS, ModelSpec, build_model, describe_models, get_spec, is_registered,
                                      list_models, model_kind, register_model, resolve_name)
from ppgbench.models import classical, nets, pretrained  # noqa: E402  (registers models on import)
from ppgbench.models.classical import (ClassicalModel, GBMFeatures, MeanRegressor, RidgeFeatures,
                                       SubjectMeanRegressor, assemble_features, predict_classical)
from ppgbench.models.nets import (CNNLSTM, FusionNet, MultimodalFusionNet, ResNet1D, SingleBranch,
                                  TinyTransformer, count_params, required_views)
from ppgbench.models.pretrained import (ENCODERS, Encoder, IdentityEncoder, PaPaGeiEncoder,
                                        PretrainedWeightsMissing, ProbeModel, RawEncoder, TorchEncoder,
                                        get_encoder)

__all__ = [
    # registry
    "KINDS", "ModelSpec", "register_model", "build_model", "list_models", "describe_models", "get_spec",
    "model_kind", "resolve_name", "is_registered",
    # classical
    "ClassicalModel", "MeanRegressor", "SubjectMeanRegressor", "RidgeFeatures", "GBMFeatures",
    "assemble_features", "predict_classical",
    # nets
    "FusionNet", "MultimodalFusionNet", "SingleBranch", "TinyTransformer", "ResNet1D", "CNNLSTM",
    "count_params", "required_views",
    # probes
    "Encoder", "IdentityEncoder", "RawEncoder", "TorchEncoder", "PaPaGeiEncoder", "PretrainedWeightsMissing",
    "ProbeModel", "ENCODERS", "get_encoder",
    # submodules
    "classical", "nets", "pretrained",
]
