"""
ppgbench.models.registry
========================
A tiny, explicit model registry.

Why a registry?  The benchmark CLI, the leakage ladder and the paper tables
all refer to models by short names (``"fusion"``, ``"1d"``, ``"ridge+demo"``).
Keeping a single name -> factory map, tagged with the model *kind*, lets the
runner decide *how* a model is evaluated (gradient training on tensor views,
scikit-learn style fitting on feature matrices, or a frozen-encoder probe)
without ``if name == ...`` chains spread across scripts.

Three kinds are recognised:

``"classical"``
    Objects with ``fit(ws, tr_idx, feats) -> self`` and
    ``predict(ws, te_idx, feats) -> np.ndarray``.  ``feats`` is a dict with
    keys ``"hand"`` (N, 9), ``"demo"`` (N, k) or ``None`` and optionally
    ``"morph"``.
``"net"``
    ``torch.nn.Module`` with ``forward(x1, x2, x3)``; unused views are ignored
    and may be passed as ``None``.  Trained by :func:`ppgbench.train.train_net`.
``"probe"``
    :class:`ppgbench.models.pretrained.ProbeModel` instances (frozen encoder +
    ridge/MLP head).  They expose the classical ``fit``/``predict`` interface
    but need the raw windows ``ws.X`` rather than precomputed features.

Usage
-----
>>> from ppgbench.models import build_model, list_models
>>> list_models()["fusion"]
'net'
>>> net = build_model("fusion")
"""
from __future__ import annotations

import difflib
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

log = logging.getLogger("ppgbench.models.registry")

KINDS: tuple[str, ...] = ("classical", "net", "probe")

#: Sub-modules that register models on import.  ``build_model`` imports them
#: lazily so that ``ppgbench.models.registry`` itself stays dependency-free.
_PROVIDER_MODULES: tuple[str, ...] = (
    "ppgbench.models.classical",
    "ppgbench.models.nets",
    "ppgbench.models.pretrained",
)


@dataclass(frozen=True)
class ModelSpec:
    """Everything the registry knows about one model name.

    Attributes
    ----------
    name:
        Canonical short name (``"fusion"``).
    kind:
        One of :data:`KINDS`.
    factory:
        Zero-or-keyword-argument callable returning a fresh model instance.
    doc:
        One-line human description shown by ``python -m ppgbench models``.
    aliases:
        Alternative names accepted by :func:`build_model` (``"subjmean"`` for
        ``"subject_mean"`` keeps the legacy ladder CLI working).
    meta:
        Free-form hints for runners, e.g. ``{"views": ("x1",)}`` tells the
        trainer which tensor views a net actually reads.
    """

    name: str
    kind: str
    factory: Callable[..., Any]
    doc: str = ""
    aliases: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ModelSpec] = {}
_ALIASES: dict[str, str] = {}
_LOADED = False


def register_model(name: str, kind: str, *, aliases: Iterable[str] = (), doc: str | None = None,
                   **meta: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering ``factory`` under ``name`` with the given ``kind``.

    Works on classes, plain functions and :func:`functools.partial` objects.
    Registering the same name twice with a *different* factory raises so that
    silent overrides cannot change benchmark results.

    Parameters
    ----------
    name:
        Canonical model name.
    kind:
        ``"classical"``, ``"net"`` or ``"probe"``.
    aliases:
        Extra names resolving to this model.
    doc:
        Description; defaults to the first docstring line of the factory.
    **meta:
        Stored verbatim in :attr:`ModelSpec.meta`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    def _decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        _register(name, kind, factory, doc=doc, aliases=tuple(aliases), meta=meta)
        return factory

    return _decorator


def _register(name: str, kind: str, factory: Callable[..., Any], *, doc: str | None,
              aliases: tuple[str, ...], meta: Mapping[str, Any]) -> ModelSpec:
    if doc is None:
        target = getattr(factory, "func", factory)          # unwrap functools.partial
        doc = (getattr(target, "__doc__", "") or "").strip().splitlines()[0:1]
        doc = doc[0].strip() if doc else ""
    spec = ModelSpec(name=name, kind=kind, factory=factory, doc=doc, aliases=aliases, meta=dict(meta))
    existing = _REGISTRY.get(name)
    if existing is not None and existing.factory is not factory:
        raise ValueError(f"model {name!r} is already registered ({existing.kind}); refusing to override")
    _REGISTRY[name] = spec
    for alias in aliases:
        if alias in _REGISTRY and alias != name:
            raise ValueError(f"alias {alias!r} for {name!r} collides with a registered model")
        _ALIASES[alias] = name
    log.debug("registered model %-16s kind=%s", name, kind)
    return spec


def _ensure_loaded() -> None:
    """Import every provider module once so their ``@register_model`` run."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for mod in _PROVIDER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:                      # pragma: no cover - surfaced in logs, not fatal
            log.warning("could not import model provider %s: %s", mod, exc)


def resolve_name(name: str) -> str:
    """Return the canonical registered name for ``name`` or an alias.

    Raises ``KeyError`` listing close matches when nothing is registered.
    """
    _ensure_loaded()
    if name in _REGISTRY:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    choices = sorted(set(_REGISTRY) | set(_ALIASES))
    hint = difflib.get_close_matches(name, choices, n=3)
    raise KeyError(f"unknown model {name!r}; " + (f"did you mean {hint}? " if hint else "") +
                   f"registered: {choices}")


def get_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name`` (aliases allowed)."""
    return _REGISTRY[resolve_name(name)]


def model_kind(name: str) -> str:
    """Return ``"classical"``, ``"net"`` or ``"probe"`` for ``name``."""
    return get_spec(name).kind


def build_model(name: str, **kw: Any) -> Any:
    """Instantiate a fresh model by name.

    ``**kw`` are forwarded to the factory (e.g. ``build_model("tiny_transformer",
    n_layers=3)``).  Every call returns a new object: nets must be re-initialised
    per fold and classical models hold fitted state.
    """
    spec = get_spec(name)
    try:
        return spec.factory(**kw)
    except TypeError as exc:
        raise TypeError(f"building model {name!r} with kwargs {sorted(kw)} failed: {exc}") from exc


def list_models(kind: str | None = None) -> dict[str, str]:
    """Return ``{name: kind}`` for every registered model (optionally filtered)."""
    _ensure_loaded()
    return {n: s.kind for n, s in sorted(_REGISTRY.items()) if kind is None or s.kind == kind}


def describe_models() -> list[dict[str, Any]]:
    """Return a table-ready list of dicts (name, kind, aliases, doc)."""
    _ensure_loaded()
    return [{"name": s.name, "kind": s.kind, "aliases": list(s.aliases), "doc": s.doc, **s.meta}
            for s in sorted(_REGISTRY.values(), key=lambda s: (s.kind, s.name))]


def is_registered(name: str) -> bool:
    """True when ``name`` (or alias) is known."""
    try:
        resolve_name(name)
        return True
    except KeyError:
        return False


__all__ = ["KINDS", "ModelSpec", "register_model", "build_model", "list_models", "describe_models",
           "get_spec", "model_kind", "resolve_name", "is_registered"]
