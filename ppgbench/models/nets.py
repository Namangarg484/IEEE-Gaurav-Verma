"""
ppgbench.models.nets
====================
Neural regressors with the uniform call signature ``forward(x1, x2, x3)``.

Views (from ``ppgbench.core.views.make_views``)::

    x1 : (B, 1, L)      z-scored 1-D PPG window (L = 320 at 64 Hz)
    x2 : (B, 1, 64, 64) CWT (mexican-hat) scalogram, min-max scaled
    x3 : (B, 3, 256)    Takens delay-embedding point cloud (tau = 5)

Models
------
``fusion``            :class:`FusionNet` - the paper's 1D+2D+3D cross-attention
                      network (thin subclass of
                      ``fusion_network_model.MultimodalFusionNet``; identical
                      parameters and initialisation order, plus ``embed``).
``1d`` / ``2d`` / ``3d``  :class:`SingleBranch` ablations: one branch + the
                      same 256->64->1 head used by ``run_subject_cv.py``.
``resnet1d`` / ``cnnlstm``  published single-channel baselines imported from
                      ``baseline_models`` (never copied, so the paper's
                      parameter counts stay authoritative).
``tiny_transformer``  :class:`TinyTransformer` - patch-16 embedding, 2 encoder
                      layers (d = 64, 4 heads), mean pool, linear head.  A
                      small attention baseline for the same 5-s windows.

The legacy modules live at the repository root; they are imported, not
vendored.  If the root is not on ``sys.path`` (e.g. tests run from elsewhere)
it is added once, derived from this file's location.

Numerical parity
----------------
:class:`SingleBranch` builds ``branch`` then ``head`` in the same order as
``run_subject_cv.SingleBranch`` so that, after ``torch.manual_seed(seed)``, the
initial weights - and hence ``train_net`` predictions - are bit-identical to
``run_subject_cv.train_fold``.
"""
from __future__ import annotations

import functools
import logging
import math
import os
import sys
from typing import Any, Iterable

import torch
import torch.nn as nn

from ppgbench.models.registry import register_model

log = logging.getLogger("ppgbench.models.nets")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_legacy() -> tuple[Any, Any]:
    """Import the root-level ``fusion_network_model`` and ``baseline_models``."""
    try:
        import baseline_models as _bm          # type: ignore
        import fusion_network_model as _fm     # type: ignore
    except ImportError:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        import baseline_models as _bm          # type: ignore
        import fusion_network_model as _fm     # type: ignore
    return _fm, _bm


_fusion_mod, _baseline_mod = _import_legacy()
Branch1D = _fusion_mod.Branch1D
Branch2D = _fusion_mod.Branch2D
Branch3D = _fusion_mod.Branch3D
CrossAttentionFusion = _fusion_mod.CrossAttentionFusion
MultimodalFusionNet = _fusion_mod.MultimodalFusionNet
ResNet1D = _baseline_mod.ResNet1D
CNNLSTM = _baseline_mod.CNNLSTM

#: Which views each net reads (used by trainers to skip unused tensors).
VIEWS_1D: tuple[str, ...] = ("x1",)
VIEWS_ALL: tuple[str, ...] = ("x1", "x2", "x3")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def required_views(name: str) -> tuple[str, ...]:
    """Views a registered net reads; defaults to all three when unknown."""
    from ppgbench.models.registry import get_spec
    return tuple(get_spec(name).meta.get("views", VIEWS_ALL))


# --------------------------------------------------------------------------- #
# Fusion network and ablations
# --------------------------------------------------------------------------- #
@register_model("fusion", "net", doc="1D+2D+3D cross-attention fusion network (paper model)", views=VIEWS_ALL)
class FusionNet(MultimodalFusionNet):
    """``MultimodalFusionNet`` plus representation helpers.

    The constructor is inherited unchanged, so parameter names, shapes and the
    initialisation sequence are exactly those of the legacy class; checkpoints
    are interchangeable.
    """

    def embed(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """Fused 768-d representation before the regression head."""
        f1, f2, f3 = self.branch_1d(x1), self.branch_2d(x2), self.branch_3d(x3)
        return self.fusion([f1, f2, f3])

    def branch_weights(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """Per-sample attention weight over the three branches, shape (B, 3)."""
        f1, f2, f3 = self.branch_1d(x1), self.branch_2d(x2), self.branch_3d(x3)
        return self.fusion.get_weights([f1, f2, f3])


class SingleBranch(nn.Module):
    """One fusion branch + the 256->64->1 regression head (ablation).

    Parameters
    ----------
    which:
        ``"1d"`` (temporal CNN on x1), ``"2d"`` (CNN on the scalogram x2) or
        ``"3d"`` (PointNet on the phase-space cloud x3).
    """

    BRANCHES = {"1d": Branch1D, "2d": Branch2D, "3d": Branch3D}

    def __init__(self, which: str) -> None:
        super().__init__()
        if which not in self.BRANCHES:
            raise ValueError(f"which must be one of {sorted(self.BRANCHES)}, got {which!r}")
        self.which = which
        self.branch = self.BRANCHES[which]()
        self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def _pick(self, x1: torch.Tensor | None, x2: torch.Tensor | None, x3: torch.Tensor | None) -> torch.Tensor:
        x = {"1d": x1, "2d": x2, "3d": x3}[self.which]
        if x is None:
            raise ValueError(f"SingleBranch({self.which!r}) needs view x{self.which[0]}, got None")
        return x

    def embed(self, x1: torch.Tensor | None = None, x2: torch.Tensor | None = None,
              x3: torch.Tensor | None = None) -> torch.Tensor:
        """256-d branch embedding."""
        return self.branch(self._pick(x1, x2, x3))

    def forward(self, x1: torch.Tensor | None = None, x2: torch.Tensor | None = None,
                x3: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x1, x2, x3))


for _w, _views in (("1d", ("x1",)), ("2d", ("x2",)), ("3d", ("x3",))):
    register_model(_w, "net", aliases=(f"branch{_w}",), views=_views,
                   doc=f"Ablation: {_w} branch of the fusion net + regression head")(
        functools.partial(SingleBranch, _w))

register_model("resnet1d", "net", doc="ResNet-1D baseline (stem + 3 residual blocks, GAP, MLP head)",
               views=VIEWS_1D)(ResNet1D)
register_model("cnnlstm", "net", doc="CNN-LSTM baseline (3 conv -> 2-layer LSTM -> MLP head)",
               views=VIEWS_1D)(CNNLSTM)


# --------------------------------------------------------------------------- #
# TinyTransformer
# --------------------------------------------------------------------------- #
@register_model("tiny_transformer", "net", aliases=("transformer", "tinytf"), views=VIEWS_1D,
                doc="Patch-16 1-D transformer: 2 encoder layers, d=64, 4 heads, mean pool")
class TinyTransformer(nn.Module):
    """A deliberately small attention model over non-overlapping 1-D patches.

    Rationale: the convolutional branches have strong locality priors; a
    transformer with learned positional embeddings tests whether *global*
    beat-to-beat relations in a 5-s window carry extra information.  At
    ~70 k parameters it is cheap enough for LOSO x seeds on CPU/MPS.

    Parameters
    ----------
    patch:
        Patch length in samples (16 = 0.25 s at 64 Hz).  Windows whose
        length is not a multiple of ``patch`` are right-padded with zeros.
    d_model, n_heads, n_layers, d_ff, dropout:
        Encoder size.  Pre-norm layers (``norm_first=True``) train stably
        with OneCycle and no warm-up tuning.
    max_len:
        Longest supported window in samples (sets the positional table size).
    """

    def __init__(self, patch: int = 16, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.patch, self.d_model = patch, d_model
        self.max_patches = math.ceil(max_len / patch)
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def embed(self, x1: torch.Tensor, x2: torch.Tensor | None = None, x3: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pooled ``d_model``-d representation of ``x1`` (B, 1, L)."""
        if x1 is None:
            raise ValueError("TinyTransformer needs view x1")
        if x1.dim() == 2:
            x1 = x1[:, None, :]
        L = x1.shape[-1]
        pad = (-L) % self.patch
        if pad:
            x1 = nn.functional.pad(x1, (0, pad))
        tok = self.patch_embed(x1).transpose(1, 2)                  # (B, P, d)
        P = tok.shape[1]
        if P > self.max_patches:
            raise ValueError(f"window of {L} samples exceeds max_len ({self.max_patches * self.patch})")
        tok = tok + self.pos_embed[:, :P]
        return self.norm(self.encoder(tok).mean(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None, x3: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x1))


# --------------------------------------------------------------------------- #
NET_NAMES: tuple[str, ...] = ("fusion", "1d", "2d", "3d", "resnet1d", "cnnlstm", "tiny_transformer")


def net_summary(names: Iterable[str] = NET_NAMES) -> dict[str, int]:
    """Parameter counts for the named nets (fresh instances)."""
    from ppgbench.models.registry import build_model
    return {n: count_params(build_model(n)) for n in names}


__all__ = ["FusionNet", "SingleBranch", "TinyTransformer", "MultimodalFusionNet", "Branch1D", "Branch2D",
           "Branch3D", "CrossAttentionFusion", "ResNet1D", "CNNLSTM", "count_params", "required_views",
           "net_summary", "NET_NAMES", "VIEWS_1D", "VIEWS_ALL"]
