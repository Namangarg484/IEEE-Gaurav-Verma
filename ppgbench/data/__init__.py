"""
ppgbench.data
=============
Datasets for the ppgbench benchmark, each loaded as a validated
:class:`~ppgbench.core.windowset.WindowSet` through a single entry point::

    from ppgbench.data import load_dataset, list_datasets, describe_datasets
    list_datasets()                     # ['bigideas', 'mendeley', 'synthetic']
    ws = load_dataset("synthetic", n_subjects=4, seed=1)

Built-in datasets
-----------------
``mendeley``   raw Mendeley fingertip PPG spot measurements (67 recordings, 23 subjects)
``bigideas``   PhysioNet BIG IDEAs wrist BVP vs Dexcom G6 CGM (16 subjects, timestamps)
``synthetic``  deterministic in-memory data for tests and protocol checks

Importing this package registers the built-in loaders; third-party loaders
register themselves with :func:`register_dataset`.
"""
from __future__ import annotations

from ppgbench.data.registry import (
    FS, HOP, WIN, DatasetError, DatasetInfo, clear_cache, dataset_info, describe_datasets,
    fingerprint, is_available, list_datasets, load_dataset, register_dataset, validate_windowset,
)
from ppgbench.data import bigideas as _bigideas  # noqa: F401  (registers "bigideas")
from ppgbench.data import mendeley as _mendeley  # noqa: F401  (registers "mendeley")
from ppgbench.data import synthetic as _synthetic  # noqa: F401  (registers "synthetic")
from ppgbench.data import stuba as _stuba  # noqa: F401  (registers "stuba")
from ppgbench.data.bigideas import fetch_bigideas, load_bigideas
from ppgbench.data.mendeley import load_mendeley
from ppgbench.data.stuba import load_stuba
from ppgbench.data.synthetic import load_synthetic, synthetic_ppg

__all__ = [
    "FS", "HOP", "WIN",
    "DatasetError", "DatasetInfo",
    "register_dataset", "load_dataset", "list_datasets", "dataset_info", "is_available",
    "describe_datasets", "validate_windowset", "fingerprint", "clear_cache",
    "load_mendeley", "load_bigideas", "fetch_bigideas", "load_stuba", "load_synthetic", "synthetic_ppg",
]
