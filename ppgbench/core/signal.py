"""
ppgbench.core.signal
====================
Signal conditioning shared by every dataset loader.  Numerically identical to
``ppg_data.py`` (``bandpass``, ``to_common_rate``, ``sliding_windows``,
``_zscore_rows``); the parity tests in ``tests/ppgbench/test_core.py`` guard
this.

Pipeline (per raw recording)
----------------------------
1. :func:`bandpass` 0.5-8 Hz, 4th-order Butterworth, zero-phase (``sosfiltfilt``)
   *at the native rate* - removes baseline wander / respiration (< 0.5 Hz) and
   high-frequency noise while keeping the cardiac fundamental (0.7-3 Hz) and
   its first harmonics that carry pulse morphology.
2. :func:`to_rate` - polyphase resampling to the common 64 Hz (the Empatica E4
   BVP rate) so that Mendeley (2175 Hz) and BIG IDEAs windows share one model.
3. :func:`sliding_windows` - fixed 5-s windows (320 samples); 50 % overlap for
   the short 10-s Mendeley recordings, no overlap for BIG IDEAs.
4. :func:`zscore_rows` - per-window z-scoring removes the device-dependent
   amplitude scale so that only waveform *shape* reaches the model.

:func:`sqi` provides per-window signal-quality indices for optional gating and
for the integrity audit.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import signal as sps
from scipy.stats import skew

log = logging.getLogger("ppgbench.core.signal")

__all__ = [
    "FS", "WIN", "HOP", "MENDELEY_FS",
    "bandpass", "to_rate", "sliding_windows", "zscore_rows",
    "sqi", "sqi_mask", "SQI_KEYS",
]

#: Common sampling rate (Hz) == Empatica E4 BVP.
FS: int = 64
#: Window length in samples (5 s).
WIN: int = 5 * FS
#: Hop between windows (50 % overlap) for the short Mendeley recordings.
HOP: int = WIN // 2
#: Native rate of the Mendeley pulse-sensor recordings.
MENDELEY_FS: int = 2175

#: Plausible resting / ambulatory heart-rate band used by :func:`sqi` (Hz).
HR_BAND: tuple[float, float] = (0.7, 3.0)


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 8.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (second-order sections).

    Parameters
    ----------
    x : 1-D array
        Raw signal; cast to float64 before filtering (as the legacy code does).
    fs : float
        Sampling rate of ``x`` in Hz.
    lo, hi : float
        Pass-band edges in Hz.  ``hi`` must be below the Nyquist frequency.
    order : int
        Butterworth order (the effective order is doubled by forward-backward
        filtering).

    Returns
    -------
    float64 array of the same length as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"bandpass expects a 1-D signal, got shape {x.shape}")
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f"need 0 < lo < hi < fs/2; got lo={lo}, hi={hi}, fs={fs}")
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def to_rate(x: np.ndarray, fs_in: float, fs_out: int = FS, lo: float = 0.5, hi: float = 8.0,
            order: int = 4) -> np.ndarray:
    """Mean-remove, band-pass at the native rate, then polyphase-resample.

    Identical to ``ppg_data.to_common_rate`` for ``fs_out=64``.  Filtering
    *before* decimation makes the band-pass double as the anti-alias filter
    (8 Hz << 32 Hz Nyquist at 64 Hz).  ``fs_in`` and ``fs_out`` must be
    integers (``resample_poly`` needs a rational ratio).

    Returns
    -------
    float32 array of length ``ceil(len(x) * fs_out / fs_in)``.
    """
    x = np.asarray(x, dtype=np.float64)
    fs_in_i, fs_out_i = int(fs_in), int(fs_out)
    if fs_in_i != fs_in or fs_out_i != fs_out:
        raise ValueError(f"to_rate needs integer rates; got fs_in={fs_in}, fs_out={fs_out}")
    x = bandpass(x - np.mean(x), fs_in_i, lo=lo, hi=hi, order=order)
    if fs_in_i == fs_out_i:
        return x.astype(np.float32)
    g = np.gcd(fs_in_i, fs_out_i)
    return sps.resample_poly(x, fs_out_i // g, fs_in_i // g).astype(np.float32)


def sliding_windows(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    """Cut a 1-D signal into ``(n, win)`` windows with stride ``hop``.

    Only complete windows are returned (the tail shorter than ``win`` is
    dropped); an empty ``(0, win)`` float32 array is returned when the signal
    is shorter than one window.  The dtype of ``x`` is preserved otherwise.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"sliding_windows expects a 1-D signal, got shape {x.shape}")
    if win <= 0 or hop <= 0:
        raise ValueError(f"win and hop must be positive; got win={win}, hop={hop}")
    n = 1 + (len(x) - win) // hop if len(x) >= win else 0
    if n == 0:
        return np.empty((0, win), np.float32)
    return np.stack([x[i * hop:i * hop + win] for i in range(n)])


def zscore_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each row (window) independently; returns float32.

    Uses the population standard deviation (``ddof=0``) plus ``eps`` so a
    flat window maps to zeros instead of NaN - identical to
    ``ppg_data._zscore_rows``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"zscore_rows expects (N, L); got shape {X.shape}")
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True) + eps
    return ((X - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------- #
# Signal quality
# --------------------------------------------------------------------------- #
#: Keys of the dict returned by :func:`sqi`.
SQI_KEYS: tuple[str, ...] = (
    "std", "flatline_frac", "dominant_freq_hz", "hr_plausible", "hr_band_power_frac",
    "skewness", "clipping_frac", "finite",
)


def sqi(X: np.ndarray, fs: int = FS, flat_tol: float = 1e-6, hr_band: tuple[float, float] = HR_BAND,
        nperseg: int | None = None) -> dict[str, np.ndarray]:
    """Per-window signal-quality indices.

    Parameters
    ----------
    X : (N, L) array
        Windows, raw or z-scored.  ``std`` is only informative on *raw*
        windows (z-scored rows have ``std == 1`` by construction); the other
        indices are scale-invariant.
    fs : int
        Sampling rate in Hz.
    flat_tol : float
        Two consecutive samples closer than ``flat_tol * peak-to-peak`` count
        as a flat segment.
    hr_band : (lo, hi)
        Band (Hz) in which the dominant frequency must fall to be plausible
        for a cardiac pulse (default 0.7-3 Hz == 42-180 bpm).

    Returns
    -------
    dict with float64 / bool arrays of length N:

    ``std``                 population standard deviation of the window
    ``flatline_frac``       fraction of consecutive sample pairs that are flat
    ``dominant_freq_hz``    Welch-spectrum peak in 0.3-10 Hz (0 if undefined)
    ``hr_plausible``        ``hr_band[0] <= dominant_freq_hz <= hr_band[1]``
    ``hr_band_power_frac``  fraction of total Welch power inside ``hr_band``
    ``skewness``            sample skewness (PPG pulses are typically right-skewed;
                            strongly negative values indicate inverted / corrupt
                            signals)
    ``clipping_frac``       fraction of samples sitting at the window min or max
                            (saturation / rail hits)
    ``finite``              all samples finite
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"sqi expects (N, L); got shape {X.shape}")
    n, L = X.shape
    if n == 0:
        return {k: np.empty(0, dtype=bool if k in ("hr_plausible", "finite") else np.float64) for k in SQI_KEYS}

    finite = np.isfinite(X).all(1)
    Xf = np.where(np.isfinite(X), X, 0.0)

    std = Xf.std(1)
    ptp = Xf.max(1) - Xf.min(1)
    d = np.abs(np.diff(Xf, axis=1))
    flat = (d <= flat_tol * np.maximum(ptp, 1e-12)[:, None]).mean(1) if L > 1 else np.ones(n)

    nps = nperseg or min(256, L)
    f, P = sps.welch(Xf, fs=fs, nperseg=min(nps, L), axis=1)
    search = (f >= 0.3) & (f <= 10.0)
    if search.any():
        Ps = P[:, search]
        dom = f[search][Ps.argmax(1)]
        dom = np.where(Ps.max(1) > 0, dom, 0.0)
    else:  # pragma: no cover - only for absurdly short windows
        dom = np.zeros(n)
    tot = P.sum(1) + 1e-12
    hr_sel = (f >= hr_band[0]) & (f <= hr_band[1])
    hr_frac = P[:, hr_sel].sum(1) / tot
    plausible = (dom >= hr_band[0]) & (dom <= hr_band[1]) & finite & (std > 0)

    sk = np.nan_to_num(skew(Xf, axis=1), nan=0.0)
    at_rail = (np.isclose(Xf, Xf.max(1, keepdims=True)) | np.isclose(Xf, Xf.min(1, keepdims=True))).mean(1)

    return {
        "std": std, "flatline_frac": flat, "dominant_freq_hz": dom, "hr_plausible": plausible,
        "hr_band_power_frac": hr_frac, "skewness": sk, "clipping_frac": at_rail, "finite": finite,
    }


def sqi_mask(q: dict[str, np.ndarray], min_std: float = 1e-3, max_flatline_frac: float = 0.2,
             require_hr_plausible: bool = True, max_clipping_frac: float = 0.1,
             skew_range: tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    """Combine :func:`sqi` indices into a boolean keep-mask.

    Defaults are deliberately lenient: the benchmark's reference results use
    *no* quality gating beyond ``std > 1e-3`` and finiteness (as in
    ``ppg_data.load_bigideas``), so gating is an opt-in ablation.
    """
    keep = q["finite"] & (q["std"] > min_std) & (q["flatline_frac"] <= max_flatline_frac)
    keep &= q["clipping_frac"] <= max_clipping_frac
    keep &= (q["skewness"] >= skew_range[0]) & (q["skewness"] <= skew_range[1])
    if require_hr_plausible:
        keep &= q["hr_plausible"]
    return np.asarray(keep, dtype=bool)


def sqi_summary(q: dict[str, np.ndarray]) -> dict[str, Any]:
    """JSON-able aggregate of :func:`sqi` output for reports / audits."""
    n = int(len(q["std"]))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "frac_hr_plausible": float(np.mean(q["hr_plausible"])),
        "frac_flat_gt_20pct": float(np.mean(q["flatline_frac"] > 0.2)),
        "frac_nonfinite": float(1.0 - np.mean(q["finite"])),
        "median_dominant_freq_hz": float(np.median(q["dominant_freq_hz"])),
        "median_hr_band_power_frac": float(np.median(q["hr_band_power_frac"])),
        "median_skewness": float(np.median(q["skewness"])),
    }
