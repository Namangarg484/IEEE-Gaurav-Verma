"""
ppgbench
========
A leakage-controlled benchmark framework for regressing a physiological target
(blood glucose, mg/dL) from wearable optical signals (PPG / BVP).

Every result the framework produces is

1. evaluated under an explicit, *named* split protocol with a machine-checked
   no-leak guarantee (:mod:`ppgbench.protocols`, :mod:`ppgbench.integrity`);
2. reported at the level of independent reference measurements, never at the
   level of overlapping windows (:func:`ppgbench.metrics.record_level`);
3. accompanied by null and classical baselines and cluster-bootstrap
   confidence intervals (:mod:`ppgbench.models.classical`,
   :func:`ppgbench.metrics.cluster_bootstrap`);
4. exportable to LaTeX / JSON / Markdown so that a manuscript never carries a
   hand-typed number (:mod:`ppgbench.report`).

The public entry points are re-exported lazily so that ``import ppgbench`` is
cheap (no torch / cv2 import until a function that needs them is touched)::

    import ppgbench
    ws    = ppgbench.load_dataset("synthetic", n_subjects=4)
    folds = ppgbench.make_folds(ws, "subject_loso")
    model = ppgbench.build_model("ridge")

Sub-packages
------------
``ppgbench.core``      signal conditioning, windowing, model views, features
``ppgbench.data``      dataset registry and loaders (Mendeley, BIG IDEAs, synthetic)
``ppgbench.models``    classical, neural and probe models behind one registry
``ppgbench.protocols`` split protocols and leak assertions
``ppgbench.metrics``   record-level metrics, Clarke / Parkes grids, bootstrap CIs
``ppgbench.integrity`` label-cycling, duplicate-window and timestamp audits
``ppgbench.report``    result store, LaTeX macros, figures
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# name -> (module, attribute).  Resolved on first access via __getattr__ so that
# ``import ppgbench`` never pulls in torch, cv2 or matplotlib.
_LAZY: dict[str, tuple[str, str]] = {
    # datasets
    "load_dataset": ("ppgbench.data.registry", "load_dataset"),
    "list_datasets": ("ppgbench.data.registry", "list_datasets"),
    "register_dataset": ("ppgbench.data.registry", "register_dataset"),
    # protocols
    "make_folds": ("ppgbench.protocols", "make_folds"),
    "leak_report": ("ppgbench.protocols", "leak_report"),
    "assert_no_leak": ("ppgbench.protocols", "assert_no_leak"),
    "LeakError": ("ppgbench.protocols", "LeakError"),
    "PROTOCOLS": ("ppgbench.protocols", "PROTOCOLS"),
    # models / training
    "build_model": ("ppgbench.models.registry", "build_model"),
    "list_models": ("ppgbench.models.registry", "list_models"),
    "register_model": ("ppgbench.models.registry", "register_model"),
    "train_net": ("ppgbench.train", "train_net"),
    # metrics / integrity
    "summarize": ("ppgbench.metrics", "summarize"),
    "record_level": ("ppgbench.metrics", "record_level"),
    "cluster_bootstrap": ("ppgbench.metrics", "cluster_bootstrap"),
    "audit": ("ppgbench.integrity", "audit"),
    # reporting
    "ResultStore": ("ppgbench.report", "ResultStore"),
    "to_latex_macros": ("ppgbench.report", "to_latex_macros"),
    "to_markdown": ("ppgbench.report", "to_markdown"),
    # core (cheap, but kept lazy for symmetry)
    "WindowSet": ("ppgbench.core.windowset", "WindowSet"),
    "make_views": ("ppgbench.core.views", "make_views"),
    "handcrafted_features": ("ppgbench.core.features", "handcrafted_features"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    """Resolve a lazily re-exported public name.

    Raises ``AttributeError`` (not ``ImportError``) for unknown names so that
    ``hasattr(ppgbench, "foo")`` behaves as for an ordinary module.
    """
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value          # cache: later accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from ppgbench.core.features import handcrafted_features
    from ppgbench.core.views import make_views
    from ppgbench.core.windowset import WindowSet
    from ppgbench.data.registry import list_datasets, load_dataset, register_dataset
    from ppgbench.integrity import audit
    from ppgbench.metrics import cluster_bootstrap, record_level, summarize
    from ppgbench.models.registry import build_model, list_models, register_model
    from ppgbench.protocols import PROTOCOLS, LeakError, assert_no_leak, leak_report, make_folds
    from ppgbench.report import ResultStore, to_latex_macros, to_markdown
    from ppgbench.train import train_net
