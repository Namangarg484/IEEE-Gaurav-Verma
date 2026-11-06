"""
ppgbench.data.synthetic
=======================
A deterministic, in-memory PPG-glucose dataset for tests, examples and
protocol sanity checks.

Design goals
------------
* **Bit-for-bit reproducible.**  Every random draw comes from
  ``numpy.random.SeedSequence(seed)`` spawned per subject, so
  ``load_synthetic(seed=0)`` returns the same arrays on every machine, and
  subject ``k`` is identical whether 3 or 30 subjects are requested
  (subset stability makes tests that vary ``n_subjects`` comparable).
* **Realistic structure.**  Subjects have their own glucose set-point and
  resting heart rate; records are CGM-like readings spaced ``spacing_s``
  (300 s) apart with monotone timestamps, so ``chronological`` and
  ``personal`` protocols work on it.  ``meta`` carries ``age, sex, height,
  weight`` so the ``demo`` baseline and ``WindowSet.demographics`` work too.
* **Learnable but not trivially so.**  The target is
  ``subject offset + signal_gain * (heart rate - 70) + noise``: the heart
  rate is readable from the waveform (it is feature 0 of the handcrafted
  features), the subject offset is not.  Hence subject-wise protocols can
  reach a positive but modest correlation, record-level leaks (same subject
  in train and test) inflate it, and null models have a clear MAE floor --
  exactly the hierarchy the leakage ladder is supposed to expose.
* **Same pipeline as the real data.**  The waveform is generated at 64 Hz,
  band-passed with ``ppgbench.core.signal.bandpass``, cut with
  ``sliding_windows`` and z-scored with ``zscore_rows``.

The waveform generator :func:`synthetic_ppg` is public because signal-quality
and feature tests benefit from a pulse of known rate and morphology.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ppgbench.core.signal import bandpass, sliding_windows, zscore_rows
from ppgbench.core.windowset import WindowSet
from ppgbench.data.registry import FS, HOP, WIN, DatasetError, register_dataset

log = logging.getLogger("ppgbench.data.synthetic")

__all__ = ["synthetic_ppg", "load_synthetic", "DEFAULT_T0"]

DEFAULT_T0: float = 1_600_000_000.0       #: 2020-09-13 12:26:40 UTC, start of subject 0
_REF_HR: float = 70.0                       #: heart rate at which the signal term is zero
_SEXES = ("Female", "Male")


def synthetic_ppg(n: int, fs: int, hr_bpm: float, rng: np.random.Generator, *,
                  resp_bpm: float = 15.0, hrv: float = 0.03, harmonic: float = 0.45,
                  notch: float = 0.25, baseline_wander: float = 0.3, noise_sd: float = 0.05) -> np.ndarray:
    """Generate ``n`` samples of a plausible PPG pulse train at ``fs`` Hz.

    The pulse is a fundamental at ``hr_bpm`` with a second harmonic (systolic
    upstroke sharpness) and a phase-shifted third harmonic standing in for the
    dicrotic notch, modulated by respiration (``resp_bpm``, amplitude and
    baseline) and beat-to-beat rate jitter (``hrv`` as a fraction).  White
    noise of ``noise_sd`` (relative to unit pulse amplitude) is added.  All
    randomness comes from ``rng``.
    """
    t = np.arange(n, dtype=np.float64) / fs
    f0 = hr_bpm / 60.0
    jitter = rng.normal(0.0, hrv * f0, size=n)
    phase = 2 * np.pi * np.cumsum((f0 + jitter) / fs)             # instantaneous phase with HRV
    pulse = (np.sin(phase) + harmonic * np.sin(2 * phase + 0.6) + notch * np.sin(3 * phase + 1.9))
    resp = 2 * np.pi * resp_bpm / 60.0 * t
    am = 1.0 + 0.15 * np.sin(resp)                                   # respiratory amplitude modulation
    wander = baseline_wander * np.sin(resp + rng.uniform(0, 2 * np.pi))
    return (am * pulse + wander + rng.normal(0.0, noise_sd, size=n)).astype(np.float64)


@register_dataset(
    "synthetic",
    description="Deterministic synthetic PPG with subject set-points, heart-rate-dependent glucose and "
                "300-s CGM-like timestamps; for tests and protocol checks only.",
    requires_files=False,
    tags=("synthetic", "cgm-like", "deterministic"),
)
def load_synthetic(n_subjects: int = 6, records_per_subject: int = 48, windows_per_record: int = 4,
                   seed: int = 0, *, fs: int = FS, win: int = WIN, hop: int = HOP, spacing_s: float = 300.0,
                   signal_gain: float = 1.2, between_subject_sd: float = 35.0, label_noise_sd: float = 3.0,
                   hr_drift_bpm: float = 6.0, noise_sd: float = 0.05, t0: float = DEFAULT_T0,
                   subject_gap_s: float = 30 * 86400.0) -> WindowSet:
    """Build a deterministic synthetic ``WindowSet``.

    Parameters
    ----------
    n_subjects, records_per_subject, windows_per_record : dataset size.  Each
        record is one reference measurement with ``windows_per_record``
        overlapping windows cut from one continuous waveform.  The default
        48 records x 300 s span 4 h per subject, so the ``chronological``
        protocol's 1-h guard gap still leaves ~70 % x 4 h - 1 h of training
        data and ``personal`` keeps > 64 training windows per subject;
        shorter sets (tests) should use ``chronological_nogap`` or a smaller
        ``gap_s``.
    seed : master seed; identical inputs give identical outputs.
    fs, win, hop : sampling rate and window geometry (defaults match the
        real data: 64 Hz, 5-s windows, 50 % overlap).
    spacing_s : seconds between consecutive records of a subject (CGM-like 300 s).
    signal_gain : mg/dL per bpm of heart-rate deviation from 70 -- the part of
        the target a model can learn from the waveform.
    between_subject_sd : SD of the subject set-points around 120 mg/dL -- the
        part a subject-wise protocol cannot learn.
    label_noise_sd : irreducible reference noise (mg/dL).
    hr_drift_bpm : amplitude of the slow within-subject heart-rate drift that
        gives chronological protocols a trend to track.
    noise_sd : additive waveform noise (relative to unit pulse amplitude).
    t0, subject_gap_s : first timestamp of subject 0 and the offset between
        consecutive subjects' start times (so subjects never overlap in time).

    Returns
    -------
    WindowSet with ``time`` monotone per subject at ``spacing_s`` spacing and
    ``meta`` columns ``record, subject, glucose, egv_time, hr_bpm,
    subject_offset, age, sex, gender, height, weight, n_windows``.
    """
    if n_subjects < 1 or records_per_subject < 1 or windows_per_record < 1:
        raise DatasetError("n_subjects, records_per_subject and windows_per_record must be >= 1")
    if win < 2 or hop < 1 or fs < 1:
        raise DatasetError("win >= 2, hop >= 1 and fs >= 1 are required")
    n_samples = (windows_per_record - 1) * hop + win
    children = np.random.SeedSequence(seed).spawn(n_subjects)
    Xs, ys, subj, rec, times, meta = [], [], [], [], [], []
    for s, child in enumerate(children):
        rng = np.random.default_rng(child)
        sid = f"S{s:02d}"
        offset = float(np.clip(rng.normal(120.0, between_subject_sd), 70.0, 250.0))
        hr_rest = float(rng.uniform(55.0, 90.0))
        age = float(rng.integers(20, 66))
        sex = _SEXES[int(rng.integers(0, 2))]
        height = float(np.round(rng.normal(170.0, 9.0), 1))
        weight = float(np.round(rng.normal(70.0, 12.0), 1))
        drift_phase = float(rng.uniform(0, 2 * np.pi))
        t_start = t0 + s * subject_gap_s
        for i in range(records_per_subject):
            hr = hr_rest + hr_drift_bpm * np.sin(2 * np.pi * i / max(records_per_subject, 2) + drift_phase) \
                 + rng.normal(0.0, 3.0)
            hr = float(np.clip(hr, 40.0, 180.0))
            glucose = offset + signal_gain * (hr - _REF_HR) + rng.normal(0.0, label_noise_sd)
            glucose = float(np.clip(np.round(glucose, 1), 40.0, 400.0))
            raw = synthetic_ppg(n_samples, fs, hr, rng, noise_sd=noise_sd)
            W = sliding_windows(bandpass(raw, fs), win, hop)
            key = f"{sid}_{i:04d}"
            t_rec = t_start + i * spacing_s
            Xs.append(W.astype(np.float32))
            ys.append(np.full(len(W), glucose, np.float32))
            subj.append(np.full(len(W), sid))
            rec.append(np.full(len(W), key))
            times.append(np.full(len(W), t_rec, np.float64))
            meta.append({"record": key, "subject": sid, "glucose": glucose, "egv_time": t_rec,
                         "hr_bpm": round(hr, 2), "subject_offset": round(offset, 2), "age": age,
                         "sex": sex, "gender": sex, "height": height, "weight": weight,
                         "n_windows": int(len(W))})
    ws = WindowSet(X=zscore_rows(np.concatenate(Xs)), y=np.concatenate(ys),
                   subject=np.concatenate(subj), record=np.concatenate(rec),
                   time=np.concatenate(times), meta=pd.DataFrame(meta), fs=int(fs), source="synthetic")
    log.info(ws.describe())
    return ws
