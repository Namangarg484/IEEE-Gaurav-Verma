"""
Shared pytest configuration for ``tests/ppgbench``.

* Puts the repository root (``ROOT``) first on ``sys.path`` so that both the
  ``ppgbench`` package and the legacy reference scripts (``ppg_data.py`` ...)
  import without installation.
* Registers the ``slow`` marker: numeric-parity tests against the real
  datasets are skipped unless ``PPGBENCH_SLOW=1``.
* Provides deterministic synthetic PPG fixtures used by every test module.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SLOW = os.environ.get("PPGBENCH_SLOW", "") == "1"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: parity tests against real datasets; run with PPGBENCH_SLOW=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if SLOW:
        return
    skip = pytest.mark.skip(reason="set PPGBENCH_SLOW=1 to run slow parity tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def synthetic_ppg(n_samples: int, fs: float, hr_hz: float = 1.2, seed: int = 0,
                  noise: float = 0.05, drift: float = 0.3) -> np.ndarray:
    """Deterministic PPG-like trace: fundamental + 2 harmonics + drift + noise.

    The second harmonic is phase-shifted so that each pulse has an
    asymmetric, right-skewed shape with a shoulder (a crude dicrotic notch).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    x = (np.sin(2 * np.pi * hr_hz * t)
         + 0.35 * np.sin(2 * np.pi * 2 * hr_hz * t + 1.2)
         + 0.10 * np.sin(2 * np.pi * 3 * hr_hz * t + 0.4)
         + drift * np.sin(2 * np.pi * 0.15 * t)
         + noise * rng.standard_normal(n_samples))
    return (1000.0 + 300.0 * x).astype(np.float64)


@pytest.fixture(scope="session")
def root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def raw_mendeley_like() -> np.ndarray:
    """10 s of 2175 Hz pulse-sensor-like signal (one Mendeley recording)."""
    return synthetic_ppg(21_900, 2175.0, hr_hz=1.1, seed=1)


@pytest.fixture(scope="session")
def windows64() -> np.ndarray:
    """(12, 320) z-scored windows at 64 Hz with varied heart rates."""
    from ppgbench.core.signal import FS, WIN, sliding_windows, to_rate, zscore_rows

    rows = []
    for k, hr in enumerate((0.9, 1.2, 1.6, 2.1)):
        x = to_rate(synthetic_ppg(2175 * 10, 2175.0, hr_hz=hr, seed=10 + k), 2175, FS)
        rows.append(sliding_windows(x, WIN, WIN // 2)[:3])
    return zscore_rows(np.concatenate(rows))
