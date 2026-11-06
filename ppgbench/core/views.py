"""
ppgbench.core.views
===================
The three complementary *views* of a PPG window consumed by the fusion
network (``fusion_network_model.py``) and its single-branch ablations:

``x1`` (N, 1, L)      the z-scored 1-D trace - waveform morphology in time;
``x2`` (N, 1, 64, 64) a continuous-wavelet (Mexican-hat) scalogram, 64 scales
                      resized to 64x64 - time-frequency energy distribution;
``x3`` (N, 3, 256)    a Takens delay-embedded point cloud (tau = 5 samples,
                      3-D) - the attractor geometry of the pulse cycle.

All three are computed once per dataset by :func:`make_views`; protocols then
only index the returned tensors.  Every function is numerically identical to
``ppg_data.py`` (``cwt_scalogram``, ``phase_space``, ``make_views``).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

log = logging.getLogger("ppgbench.core.views")

__all__ = [
    "TAU", "PHASE_POINTS", "CWT_SCALES", "SPEC", "VIEW_KEYS",
    "cwt_scalogram", "phase_space", "make_views", "make_views_numpy", "view_shapes",
]

#: Takens embedding delay in samples (78 ms at 64 Hz).
TAU: int = 5
#: Number of points kept in the phase-space cloud.
PHASE_POINTS: int = 256
#: Number of CWT scales (1..64).
CWT_SCALES: int = 64
#: Scalogram side length in pixels after resizing.
SPEC: int = 64
#: Keys of the dict returned by :func:`make_views`, in branch order.
VIEW_KEYS: tuple[str, str, str] = ("x1", "x2", "x3")


def cwt_scalogram(x: np.ndarray, n_scales: int = CWT_SCALES, size: int = SPEC,
                  wavelet: str = "mexh") -> np.ndarray:
    """Min-max-normalised ``(size, size)`` float32 scalogram of one window.

    Scales ``1..n_scales`` of the Mexican-hat wavelet are computed with
    :func:`pywt.cwt`, the magnitude is bilinearly resized with OpenCV to
    ``size x size`` and scaled to ``[0, 1]``.  A constant scalogram (dynamic
    range below 1e-8) is returned un-normalised, exactly as the legacy code.
    """
    import cv2
    import pywt

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"cwt_scalogram expects a 1-D window, got shape {x.shape}")
    coef, _ = pywt.cwt(x, np.arange(1, n_scales + 1, dtype=np.float64), wavelet)
    mag = cv2.resize(np.abs(coef).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    mn, mx = mag.min(), mag.max()
    return (mag - mn) / (mx - mn) if mx - mn > 1e-8 else mag


def phase_space(x: np.ndarray, tau: int = TAU, n_points: int = PHASE_POINTS) -> np.ndarray:
    """3-D Takens delay embedding ``(3, n_points)`` float32.

    Rows are ``x(t), x(t+tau), x(t+2tau)`` for ``t`` in ``[0, L - 2 tau)``.
    When more than ``n_points`` embedded points exist they are sub-sampled
    with ``linspace(...).round()`` so that the whole window is covered;
    shorter windows are zero-padded on the right.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"phase_space expects a 1-D window, got shape {x.shape}")
    L = len(x) - 2 * tau
    if L <= 0:
        return np.zeros((3, n_points), np.float32)
    pts = np.vstack((x[:L], x[tau:tau + L], x[2 * tau:2 * tau + L])).astype(np.float32)
    if pts.shape[1] >= n_points:
        idx = np.linspace(0, pts.shape[1] - 1, n_points).round().astype(int)
        return pts[:, idx]
    return np.pad(pts, ((0, 0), (0, n_points - pts.shape[1])))


def make_views_numpy(X: np.ndarray, n_scales: int = CWT_SCALES, size: int = SPEC, tau: int = TAU,
                     n_points: int = PHASE_POINTS) -> dict[str, np.ndarray]:
    """Same as :func:`make_views` but returns float32 numpy arrays (no torch)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"make_views expects (N, L); got shape {X.shape}")
    n, L = X.shape
    if n == 0:
        return {"x1": np.empty((0, 1, L), np.float32), "x2": np.empty((0, 1, size, size), np.float32),
                "x3": np.empty((0, 3, n_points), np.float32)}
    x2 = np.stack([cwt_scalogram(x, n_scales, size) for x in X])[:, None]
    x3 = np.stack([phase_space(x, tau, n_points) for x in X])
    return {"x1": X[:, None, :].astype(np.float32), "x2": x2.astype(np.float32), "x3": x3.astype(np.float32)}


def make_views(X: np.ndarray, **kw: int) -> dict[str, "torch.Tensor"]:
    """Return ``{'x1': (N,1,L), 'x2': (N,1,64,64), 'x3': (N,3,256)}`` float32 tensors.

    Tensors live on the CPU; callers move mini-batches to the training device.
    Keyword arguments are forwarded to :func:`make_views_numpy`.
    """
    import torch

    views = make_views_numpy(X, **kw)
    log.debug("views for %d windows: %s", len(X), {k: v.shape for k, v in views.items()})
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in views.items()}


def view_shapes(L: int = 320, size: int = SPEC, n_points: int = PHASE_POINTS) -> dict[str, tuple[int, ...]]:
    """Per-sample shapes of the three views (without the batch axis)."""
    return {"x1": (1, L), "x2": (1, size, size), "x3": (3, n_points)}
