"""
ppgbench.core.features
======================
Per-window descriptors for the classical baselines and the probe models.

* :func:`handcrafted_features` - nine cheap, fully vectorised spectral and
  statistical descriptors, numerically identical to
  ``ppg_data.handcrafted_features``.  They are the *reference* feature set:
  the ridge / GBM baselines in the paper use exactly these.
* :func:`morphology_features` - beat-level pulse-morphology statistics
  (rise time, pulse width, dicrotic-notch position / depth, inter-beat
  variability) that require a peak detector.  Peak detection comes from
  ``neurokit2`` (optional dependency, guarded import); the per-beat geometry is
  computed here in numpy so that results do not depend on neurokit internals
  beyond the peak indices.
* :func:`feature_bundle` - builds the ``feats`` dict that classical models
  consume: ``{"hand": (N,9), "demo": (N,k) | None, "morph": (N,m) | absent}``.

Scientific rationale
--------------------
Glucose has no established direct signature in a single PPG pulse; published
claims rest on indirect correlates (heart rate and its variability, vascular
compliance seen in pulse width / dicrotic-notch timing, and spectral energy
redistribution).  The hand-crafted set therefore spans heart rate, the energy
split across the cardiac fundamental / harmonic / high-frequency bands, total
log-power, shape moments (skewness, kurtosis) and two roughness measures.
Any model that cannot beat these features under a subject-wise protocol has
learned identity, not physiology.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, skew

from ppgbench.core.signal import FS

if TYPE_CHECKING:  # pragma: no cover
    from ppgbench.core.windowset import WindowSet

log = logging.getLogger("ppgbench.core.features")

__all__ = [
    "FEATURE_NAMES", "MORPHOLOGY_FEATURE_NAMES", "DEMOGRAPHIC_NAMES",
    "handcrafted_features", "morphology_features", "feature_bundle", "neurokit_available",
]

#: Column names of :func:`handcrafted_features` (order is binding).
FEATURE_NAMES: tuple[str, ...] = (
    "hr_bpm",            # Welch peak in 0.7-3 Hz, x60
    "band_0p5_1p5",      # fraction of power in 0.5-1.5 Hz (cardiac fundamental)
    "band_1p5_3",        # fraction of power in 1.5-3 Hz (first harmonic / fast HR)
    "band_3_8",          # fraction of power in 3-8 Hz (higher harmonics, notch detail)
    "log_total_power",   # log of total Welch power
    "skewness",          # sample skewness of the window
    "kurtosis",          # excess kurtosis of the window
    "mean_abs_diff",     # mean absolute first difference (roughness)
    "zero_cross_diff",   # fraction of sign changes of the first difference (local extrema rate)
)

#: Column names of :func:`morphology_features` (order is binding).
MORPHOLOGY_FEATURE_NAMES: tuple[str, ...] = (
    "n_beats",                 # systolic peaks detected in the window
    "ibi_mean_s",              # mean inter-beat interval (s)
    "ibi_sdnn_s",              # SD of inter-beat intervals (s)
    "ibi_rmssd_s",             # RMS of successive IBI differences (s)
    "rise_time_s",             # mean foot-to-peak time (s)
    "decay_time_s",            # mean peak-to-next-foot time (s)
    "pulse_width_half_s",      # mean width at half of the pulse amplitude (s)
    "amp_mean",                # mean foot-to-peak amplitude (z-units)
    "amp_cv",                  # coefficient of variation of the amplitude
    "notch_frac",              # fraction of beats with a detectable dicrotic notch
    "notch_rel_time",          # mean notch position as fraction of the beat (0-1)
    "notch_rel_depth",         # mean (peak - notch) / amplitude
    "reflection_index",        # mean diastolic-peak / systolic-peak amplitude ratio
    "sys_frac",                # mean rise_time / IBI (systolic fraction)
)

#: Names of the default demographic columns produced by :meth:`WindowSet.demographics`.
DEMOGRAPHIC_NAMES: tuple[str, ...] = ("age", "height", "weight", "male")


# --------------------------------------------------------------------------- #
# Hand-crafted spectral / statistical features (legacy-identical)
# --------------------------------------------------------------------------- #
def handcrafted_features(X: np.ndarray, fs: int = FS) -> np.ndarray:
    """Per-window PPG descriptors: HR, spectral bands, morphology moments.

    Parameters
    ----------
    X : (N, L) array
        Windows (z-scored in the benchmark, but any scale works; only
        ``log_total_power`` is scale dependent).
    fs : int
        Sampling rate in Hz (Welch ``nperseg = min(256, L)``).

    Returns
    -------
    (N, 9) float32 array in :data:`FEATURE_NAMES` order with NaN/inf replaced
    by 0 - identical to ``ppg_data.handcrafted_features`` for ``fs=64``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"handcrafted_features expects (N, L); got shape {X.shape}")
    if X.shape[0] == 0:
        return np.empty((0, len(FEATURE_NAMES)), np.float32)
    f, P = sps.welch(X, fs=fs, nperseg=min(256, X.shape[1]), axis=1)

    def band(lo: float, hi: float) -> np.ndarray:
        return P[:, (f >= lo) & (f < hi)].sum(1)

    tot = P.sum(1) + 1e-12
    hr_band = (f >= 0.7) & (f <= 3.0)
    hr = f[hr_band][P[:, hr_band].argmax(1)] * 60.0
    feats = np.column_stack([
        hr, band(0.5, 1.5) / tot, band(1.5, 3.0) / tot, band(3.0, 8.0) / tot,
        np.log(tot), skew(X, axis=1), kurtosis(X, axis=1),
        np.abs(np.diff(X, axis=1)).mean(1), (np.diff(np.sign(np.diff(X, axis=1)), axis=1) != 0).mean(1),
    ])
    return np.nan_to_num(feats).astype(np.float32)


# --------------------------------------------------------------------------- #
# Pulse morphology (optional neurokit2 peak detection)
# --------------------------------------------------------------------------- #
_NK_MISSING_MSG = (
    "morphology_features needs the optional dependency 'neurokit2' "
    "(pip install neurokit2) for systolic-peak detection; it is not importable "
    "in this interpreter.  Use handcrafted_features instead or install it."
)


def neurokit_available() -> bool:
    """True when ``neurokit2`` can be imported."""
    try:
        import neurokit2  # noqa: F401
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


def _find_peaks_nk(x: np.ndarray, fs: int, method: str) -> np.ndarray:
    """Systolic peak indices via neurokit2 (raises ImportError if unavailable)."""
    try:
        import neurokit2 as nk
    except ImportError as e:
        raise ImportError(_NK_MISSING_MSG) from e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = nk.ppg_findpeaks(np.asarray(x, dtype=np.float64), sampling_rate=fs, method=method)
    return np.asarray(info["PPG_Peaks"], dtype=int)


def _beat_morphology(x: np.ndarray, peaks: np.ndarray, fs: int) -> np.ndarray:
    """Numpy geometry of each beat between consecutive systolic peaks.

    For each pair of peaks ``(p_k, p_{k+1})`` the *foot* of beat ``k+1`` is the
    minimum between them; the dicrotic notch is searched as a local minimum of
    the signal between the systolic peak and the foot (after the first 15 % of
    the beat), the diastolic peak as the local maximum after the notch.
    """
    L = len(x)
    nb = len(peaks)
    out = np.full(len(MORPHOLOGY_FEATURE_NAMES), np.nan)
    out[0] = nb
    if nb < 2:
        return out
    ibi = np.diff(peaks) / fs
    out[1] = ibi.mean()
    out[2] = ibi.std(ddof=1) if nb > 2 else 0.0
    out[3] = np.sqrt(np.mean(np.diff(ibi) ** 2)) if nb > 2 else 0.0

    rise, decay, width, amp, notch_t, notch_d, refl, sysf = [], [], [], [], [], [], [], []
    notch_found = 0
    n_beats_geom = 0
    for k in range(nb - 1):
        p0, p1 = int(peaks[k]), int(peaks[k + 1])
        if p1 - p0 < 3:
            continue
        seg = x[p0:p1 + 1]
        foot_rel = int(np.argmin(seg))                      # foot of the next beat
        foot = p0 + foot_rel
        a0 = x[p0] - x[foot]                                # amplitude of beat k (peak to following trough)
        n_beats_geom += 1
        # rise time of beat k+1: foot -> p1
        rise.append((p1 - foot) / fs)
        decay.append((foot - p0) / fs)
        amp.append(a0)
        sysf.append((p1 - foot) / (p1 - p0))
        # width at half amplitude around p0 (uses previous foot if available)
        half = x[foot] + 0.5 * a0
        left = p0
        lo_bound = int(peaks[k - 1]) if k > 0 else 0
        while left > lo_bound and x[left] > half:
            left -= 1
        right = p0
        while right < foot and x[right] > half:
            right += 1
        width.append((right - left) / fs)
        # dicrotic notch: local minimum between peak and foot, after 15 % of the beat
        start = p0 + max(1, int(0.15 * (p1 - p0)))
        if foot - start >= 3 and a0 > 0:
            s = x[start:foot]
            d1 = np.diff(s)
            mins = np.flatnonzero((d1[:-1] < 0) & (d1[1:] >= 0)) + 1    # local minima inside (start, foot)
            if len(mins):
                m = start + int(mins[0])
                after = x[m:foot]
                dp = m + int(np.argmax(after)) if len(after) else m
                notch_found += 1
                notch_t.append((m - p0) / (p1 - p0))
                notch_d.append((x[p0] - x[m]) / a0)
                refl.append((x[dp] - x[foot]) / a0)
    if n_beats_geom:
        out[4] = np.mean(rise)
        out[5] = np.mean(decay)
        out[6] = np.mean(width)
        out[7] = np.mean(amp)
        out[8] = (np.std(amp) / np.mean(amp)) if np.mean(amp) > 0 else np.nan
        out[9] = notch_found / n_beats_geom
        out[10] = np.mean(notch_t) if notch_t else np.nan
        out[11] = np.mean(notch_d) if notch_d else np.nan
        out[12] = np.mean(refl) if refl else np.nan
        out[13] = np.mean(sysf)
    return out


def morphology_features(X: np.ndarray, fs: int = FS, method: str = "elgendi",
                        fill_value: float = 0.0, return_nan: bool = False) -> np.ndarray:
    """Beat-level pulse-morphology features ``(N, 14)`` float32.

    Systolic peaks are detected per window with ``neurokit2.ppg_findpeaks``
    (``method`` forwarded, default Elgendi 2013); rise time, pulse width,
    dicrotic-notch and inter-beat statistics are then measured in numpy
    (:data:`MORPHOLOGY_FEATURE_NAMES`).  Windows where detection fails or
    fewer than two beats are found yield NaN entries, replaced by
    ``fill_value`` unless ``return_nan=True``.

    Raises
    ------
    ImportError
        with an explicit message when ``neurokit2`` is not importable.
    """
    if not neurokit_available():
        raise ImportError(_NK_MISSING_MSG)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"morphology_features expects (N, L); got shape {X.shape}")
    out = np.full((X.shape[0], len(MORPHOLOGY_FEATURE_NAMES)), np.nan)
    n_fail = 0
    for i, x in enumerate(X):
        if not np.isfinite(x).all() or x.std() < 1e-8:
            n_fail += 1
            continue
        try:
            peaks = _find_peaks_nk(x, fs, method)
        except ImportError:
            raise
        except Exception as e:  # neurokit raises on degenerate windows
            log.debug("peak detection failed on window %d: %s", i, e)
            n_fail += 1
            out[i, 0] = 0
            continue
        out[i] = _beat_morphology(x, peaks, fs)
    if n_fail:
        log.info("morphology_features: %d/%d windows without usable beats", n_fail, len(X))
    if return_nan:
        return out.astype(np.float32)
    return np.nan_to_num(out, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature bundle for classical models
# --------------------------------------------------------------------------- #
def feature_bundle(ws: "WindowSet", include_morph: bool = False, morph_kw: dict[str, Any] | None = None,
                   demo_cols: tuple[str, ...] = ("age", "height", "weight"),
                   sex_col: str = "sex") -> dict[str, np.ndarray | None]:
    """Build the ``feats`` dict consumed by classical models.

    Returns ``{"hand": (N,9), "demo": (N,k) or None}`` plus ``"morph": (N,14)``
    when ``include_morph`` is true (requires ``neurokit2``).  ``demo`` follows
    :meth:`WindowSet.demographics` and is ``None`` for datasets without
    demographics (BIG IDEAs).
    """
    feats: dict[str, np.ndarray | None] = {
        "hand": handcrafted_features(ws.X, fs=ws.fs),
        "demo": ws.demographics(cols=demo_cols, sex_col=sex_col),
    }
    if include_morph:
        feats["morph"] = morphology_features(ws.X, fs=ws.fs, **(morph_kw or {}))
    log.debug("feature bundle: %s", {k: None if v is None else v.shape for k, v in feats.items()})
    return feats
