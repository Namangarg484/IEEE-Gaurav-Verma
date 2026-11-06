"""
ppgbench.core
=============
Dataset-agnostic building blocks shared by every loader, model and protocol:

* :mod:`~ppgbench.core.windowset` - the :class:`WindowSet` container that every
  loader returns and every protocol / model consumes;
* :mod:`~ppgbench.core.signal`    - band-pass, rate conversion, windowing,
  per-window z-scoring and signal-quality indices;
* :mod:`~ppgbench.core.views`     - the three model views (1-D trace, CWT
  scalogram, Takens phase-space point cloud);
* :mod:`~ppgbench.core.features`  - hand-crafted spectral/morphological
  descriptors and optional neurokit2 pulse-morphology features.

All functions reproduce the legacy ``ppg_data.py`` numerically; the parity
tests live in ``tests/ppgbench/test_core.py``.
"""
from __future__ import annotations

from ppgbench.core.features import (
    FEATURE_NAMES,
    MORPHOLOGY_FEATURE_NAMES,
    feature_bundle,
    handcrafted_features,
    morphology_features,
)
from ppgbench.core.signal import (
    FS,
    HOP,
    MENDELEY_FS,
    WIN,
    bandpass,
    sliding_windows,
    sqi,
    sqi_mask,
    to_rate,
    zscore_rows,
)
from ppgbench.core.views import (
    CWT_SCALES,
    PHASE_POINTS,
    SPEC,
    TAU,
    VIEW_KEYS,
    cwt_scalogram,
    make_views,
    phase_space,
    view_shapes,
)
from ppgbench.core.windowset import WindowSet, concat_windowsets

__all__ = [
    "FS", "WIN", "HOP", "MENDELEY_FS",
    "TAU", "PHASE_POINTS", "CWT_SCALES", "SPEC", "VIEW_KEYS",
    "FEATURE_NAMES", "MORPHOLOGY_FEATURE_NAMES",
    "WindowSet", "concat_windowsets",
    "bandpass", "to_rate", "sliding_windows", "zscore_rows", "sqi", "sqi_mask",
    "cwt_scalogram", "phase_space", "make_views", "view_shapes",
    "handcrafted_features", "morphology_features", "feature_bundle",
]
