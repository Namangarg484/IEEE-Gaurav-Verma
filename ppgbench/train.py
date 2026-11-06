"""
ppgbench.train
==============
Fixed-budget training of one network on one fold.

Protocol (why it looks like this)
---------------------------------
* **No peeking.**  A fresh model per fold, a fixed epoch budget, a OneCycle
  schedule: there is no early stopping, no checkpoint selection and no
  learning-rate tuning on the held-out subject.  Anything else turns the test
  fold into a validation fold and inflates every metric.
* **Target standardisation with train statistics only.**  Glucose is
  standardised by the training-fold mean/SD and predictions are
  de-standardised; the test labels are never touched.
* **Smooth-L1 on the standardised target** - robust to the few extreme
  glucose values without the gradient collapse of pure L1.
* **AdamW (wd 1e-4), OneCycle (pct_start 0.2), grad-norm clip 1.0,
  drop_last** - BatchNorm needs > 1 sample per step; dropping the ragged
  last batch keeps statistics sane and the step count deterministic.

:func:`train_net` reproduces ``run_subject_cv.train_fold`` bit-for-bit on the
same device and seed (same RNG consumption order: ``manual_seed`` -> model
init -> one ``randperm`` per epoch).  :func:`fit_net`/:func:`predict_net`
expose the two halves for callers that want the trained module (probing,
attention maps, checkpoints).
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from ppgbench.models.registry import build_model, get_spec

log = logging.getLogger("ppgbench.train")

Views = Mapping[str, torch.Tensor]
VIEW_KEYS: tuple[str, ...] = ("x1", "x2", "x3")


# --------------------------------------------------------------------------- #
# Device and seeding
# --------------------------------------------------------------------------- #
def device(prefer: str | torch.device | None = None) -> torch.device:
    """Pick the compute device.

    Order: explicit ``prefer`` > ``$PPGBENCH_DEVICE`` > MPS if available > CPU
    (the legacy default).  CUDA is honoured when requested and available.
    """
    want = str(prefer) if prefer is not None else os.environ.get("PPGBENCH_DEVICE", "")
    if want:
        if want.startswith("mps") and not torch.backends.mps.is_available():
            log.warning("MPS requested but unavailable; using CPU")
            return torch.device("cpu")
        if want.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device(want)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


get_device = device   # alias usable where the name ``device`` is shadowed


def seed_everything(seed: int) -> None:
    """Seed torch (CPU + accelerators), numpy and ``random``.

    The first two calls are exactly those of the legacy trainer, so the RNG
    stream consumed by model initialisation and ``torch.randperm`` is
    unchanged; seeding ``random`` additionally is harmless and makes any
    Python-level shuffling deterministic.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# Configuration and result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """Optimisation budget for one fold (defaults = legacy ``run_subject_cv`` values)."""

    epochs: int = 40
    batch: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pct_start: float = 0.2
    clip_grad: float = 1.0
    eval_batch: int = 256

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1 (a fixed, non-zero budget)")
        if self.batch < 2:
            raise ValueError("batch must be >= 2 (BatchNorm needs more than one sample)")


@dataclass
class FoldResult:
    """Output of :func:`train_fold`: predictions plus the trained module and bookkeeping."""

    pred: np.ndarray
    model: nn.Module
    y_mean: float
    y_std: float
    history: list[float] = field(default_factory=list)   # mean train loss per epoch (standardised units)
    seconds: float = 0.0
    n_steps: int = 0

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary without the module."""
        return {"y_mean": self.y_mean, "y_std": self.y_std, "n_steps": self.n_steps,
                "seconds": round(self.seconds, 2), "final_loss": self.history[-1] if self.history else None}


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _check_inputs(name: str, views: Views, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> None:
    spec = get_spec(name)
    if spec.kind != "net":
        raise ValueError(f"{name!r} is a {spec.kind} model; train_net only trains kind='net' models")
    n = len(y)
    needed = tuple(spec.meta.get("views", VIEW_KEYS))
    for k in needed:
        if k not in views or views[k] is None:
            raise ValueError(f"model {name!r} needs view {k!r}; views has {sorted(views)}")
        if len(views[k]) != n:
            raise ValueError(f"view {k!r} has {len(views[k])} rows but y has {n}")
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(f"empty fold: n_train={len(tr)} n_test={len(te)}")
    if tr.max() >= n or te.max() >= n or tr.min() < 0 or te.min() < 0:
        raise IndexError("fold indices out of range")


def _gather(views: Views, idx: torch.Tensor | np.ndarray, dev: torch.device) -> list[torch.Tensor | None]:
    """Index each of x1, x2, x3 (None when absent) and move to ``dev``."""
    out: list[torch.Tensor | None] = []
    for k in VIEW_KEYS:
        v = views.get(k)
        out.append(v[idx].to(dev) if v is not None else None)
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fit_net(name: str, views: Views, y: np.ndarray, tr: np.ndarray, cfg: TrainConfig, seed: int,
            dev: torch.device, **model_kw: Any) -> FoldResult:
    """Train a fresh ``name`` network on rows ``tr`` with the budget ``cfg``.

    Returns a :class:`FoldResult` whose ``pred`` is empty; call
    :func:`predict_net` for predictions.  The RNG consumption order matches
    the legacy trainer exactly.
    """
    t0 = time.time()
    y = np.asarray(y, np.float32)
    tr = np.asarray(tr, np.int64)
    seed_everything(seed)
    model = build_model(name, **model_kw).to(dev)
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-6)
    yt = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_batches = len(tr) // cfg.batch
    steps = cfg.epochs * max(1, n_batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=cfg.pct_start)
    if n_batches == 0:
        log.warning("%s: n_train=%d < batch=%d -> drop_last leaves no steps; model stays at init",
                    name, len(tr), cfg.batch)
    xs = {k: v[tr] for k, v in views.items() if v is not None}
    model.train()
    history: list[float] = []
    n_steps = 0
    for _ep in range(cfg.epochs):
        perm = torch.randperm(len(tr))
        ep_loss: list[torch.Tensor] = []
        for b in range(n_batches):                         # drop_last: BatchNorm needs > 1 sample
            idx = perm[b * cfg.batch:(b + 1) * cfg.batch]
            x1, x2, x3 = _gather(xs, idx, dev)
            pred = model(x1, x2, x3).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, yt[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            sched.step()
            ep_loss.append(loss.detach())
            n_steps += 1
        if ep_loss:
            history.append(float(torch.stack(ep_loss).mean().cpu()))
    model.eval()
    return FoldResult(pred=np.empty(0, np.float32), model=model, y_mean=mu, y_std=sd, history=history,
                      seconds=time.time() - t0, n_steps=n_steps)


def predict_net(model: nn.Module, views: Views, idx: np.ndarray, y_mean: float, y_std: float,
                dev: torch.device | None = None, batch: int = 256) -> np.ndarray:
    """De-standardised predictions (mg/dL) of ``model`` for rows ``idx``."""
    dev = dev if dev is not None else next(model.parameters()).device
    idx = np.asarray(idx, np.int64)
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            i = idx[b:b + batch]
            x1, x2, x3 = _gather(views, i, dev)
            out.append(model(x1, x2, x3).squeeze(1).cpu())
    if not out:
        return np.empty(0, np.float32)
    return (torch.cat(out).numpy() * y_std + y_mean).astype(np.float32)


def train_fold(name: str, views: Views, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               cfg: TrainConfig | None = None, seed: int = 0, dev: torch.device | str | None = None,
               **model_kw: Any) -> FoldResult:
    """Fit on ``tr`` and predict ``te``; returns the full :class:`FoldResult`."""
    cfg = cfg or TrainConfig()
    dev_ = get_device(dev) if not isinstance(dev, torch.device) else dev
    tr, te = np.asarray(tr, np.int64), np.asarray(te, np.int64)
    _check_inputs(name, views, y, tr, te)
    res = fit_net(name, views, y, tr, cfg, seed, dev_, **model_kw)
    res.pred = predict_net(res.model, views, te, res.y_mean, res.y_std, dev_, cfg.eval_batch)
    res.seconds = res.seconds + 0.0
    log.debug("%s seed %d: %d steps, %.1f s, MAE(win) %.2f", name, seed, res.n_steps, res.seconds,
              float(np.abs(res.pred - np.asarray(y)[te]).mean()))
    return res


def train_net(name: str, views: Views, y: np.ndarray, tr: np.ndarray, te: np.ndarray, epochs: int = 40,
              batch: int = 16, lr: float = 1e-3, seed: int = 0, device: torch.device | str | None = None,
              **model_kw: Any) -> np.ndarray:
    """Train ``name`` on ``tr`` and return predictions for ``te`` (mg/dL, float32).

    Drop-in replacement for ``run_subject_cv.train_fold(name, views, y, tr,
    te, epochs, batch, lr, seed, dev)``: same optimiser, schedule, loss,
    clipping, drop_last batching and target standardisation, hence identical
    numbers for the same seed and device.  ``**model_kw`` go to
    :func:`ppgbench.models.build_model`.
    """
    cfg = TrainConfig(epochs=epochs, batch=batch, lr=lr)
    return train_fold(name, views, y, tr, te, cfg, seed, device, **model_kw).pred


def config_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Plain dict of a :class:`TrainConfig` for result stores."""
    return asdict(cfg)


__all__ = ["device", "get_device", "seed_everything", "TrainConfig", "FoldResult", "fit_net", "predict_net",
           "train_fold", "train_net", "config_dict", "VIEW_KEYS"]
