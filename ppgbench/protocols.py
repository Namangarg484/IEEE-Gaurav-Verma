"""
ppgbench.protocols
==================
Named split protocols with a machine-checked no-leak guarantee.

Why this module exists
----------------------
The first draft of this project reported an MAE of ~2 mg/dL because windows of
the same recording, and labels of the same subject, sat on both sides of the
train/test split.  Every split used by ppgbench therefore

1. has a *name* (``make_folds(ws, "subject_loso")``),
2. has a *declared leakage level* -- the strongest kind of information that is
   guaranteed not to cross from train to test, and
3. is verified against that level with exact set arithmetic (NumPy
   ``intersect1d`` on integer codes) before any model sees it.

Leakage levels
--------------
``window``
    No window index is in both train and test.  This is the weakest
    guarantee: overlapping windows of the *same* recording still share the
    label and most of the signal (``window_random``, the leakiest rung of
    the leakage ladder).
``record``
    No reference measurement (a Mendeley recording, a CGM reading) is in both.
    Subjects are still shared, so a model can memorise a person's typical
    glucose (``record_random``, ``record_kfold``).  Implies ``window``.
``subject``
    No subject is in both.  The honest cross-person protocol
    (``subject_kfold``, ``subject_loso``).  Implies ``record`` and ``window``.
``time``
    Subjects *are* shared by design, but every test reading is more than
    ``gap_s`` seconds away from the *nearest* training reading of the same
    subject.  This is the within-person protocol for CGM-aligned data, where
    neighbouring readings are autocorrelated and a guard gap is needed
    (``chronological``, ``chronological_nogap``, ``chronological_blocked``,
    ``personal``).  Implies ``record`` and ``window`` (a shared record has
    distance 0) but not ``subject``.

    The framework spec phrases the rule as "no test reading inside
    ``[train_min, train_max + gap]``".  For the chronological protocols (test
    strictly after train) the two formulations are identical.  The
    nearest-reading rule is used here because (i) autocorrelation is
    symmetric, so a test reading shortly *before* the first training reading
    leaks too, and (ii) blocked time-series CV (``chronological_blocked``)
    legitimately trains on both sides of a test block; the interval rule would
    forbid it even with an arbitrarily large gap.  ``leak_report`` still
    reports how many test windows fall inside the training span
    (``test_in_train_span_windows``: interpolation vs. extrapolation).

Legacy parity
-------------
The builders reproduce, index for index,

* ``run_subject_cv.py``        LeaveOneGroupOut / GroupKFold by subject,
* ``run_leakage_ladder.py``    ``mendeley_folds`` (window / record / subject)
  and ``bigideas_folds`` (reading / chrono0 / chrono),
* ``run_bigideas_cv.py``       ``chronological_split`` (quantile over the
  unique reading times of each subject, guard gap) and the ``personal``
  protocol (one fold per subject, subjects with <= 64 training windows
  skipped).

The legacy protocol names are accepted as aliases (``LEGACY_ALIASES``).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

if TYPE_CHECKING:  # pragma: no cover - the WindowSet is duck-typed at runtime
    from ppgbench.core.windowset import WindowSet

__all__ = [
    "Fold",
    "LEVELS",
    "LeakError",
    "ProtocolSpec",
    "PROTOCOLS",
    "LEGACY_ALIASES",
    "register_protocol",
    "resolve_protocol",
    "list_protocols",
    "declared_level",
    "protocol_variant",
    "split_protocol_variant",
    "make_folds",
    "verify_folds",
    "assert_no_leak",
    "leak_report",
    "window_time",
    "chronological_split",
    "window_random",
    "record_random",
    "record_kfold",
    "subject_kfold",
    "subject_loso",
    "chronological",
    "chronological_nogap",
    "chronological_blocked",
    "personal",
    "fold_fingerprint",
    "folds_to_json",
    "folds_from_json",
    "describe_folds",
]

log = logging.getLogger("ppgbench.protocols")

Fold = tuple[np.ndarray, np.ndarray]
"""``(train_idx, test_idx)``: sorted ``int64`` window indices into a WindowSet."""

LEVELS: tuple[str, ...] = ("window", "record", "subject", "time")
"""Leakage levels, weakest first (``time`` is a sibling of ``subject``)."""

# Which elementary checks a level requires.  Order = order of reporting.
_CHECKS_FOR_LEVEL: dict[str, tuple[str, ...]] = {
    "window": ("window",),
    "record": ("window", "record"),
    "subject": ("window", "record", "subject"),
    "time": ("window", "record", "time"),
}

_N_EXAMPLES = 5


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LeakError(AssertionError):
    """A fold lets information cross from train to test at the asserted level.

    Attributes
    ----------
    level : str
        The level that was asserted (``window`` / ``record`` / ``subject`` /
        ``time``).
    violated : str
        The elementary check that failed first (same vocabulary as ``level``).
    fold : int | None
        0-based index of the offending fold.
    protocol : str | None
        Protocol name, when known.
    n_shared : int
        Number of shared keys (windows / records / subjects) or, for ``time``,
        the number of test windows inside the forbidden interval.
    examples : tuple
        Up to five offending keys for the error message.
    counts : dict
        The full per-fold count dictionary (see :func:`leak_report`).
    """

    def __init__(self, message: str, *, level: str, violated: str, fold: int | None = None,
                 protocol: str | None = None, n_shared: int = 0, examples: Sequence[Any] = (),
                 counts: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.level = level
        self.violated = violated
        self.fold = fold
        self.protocol = protocol
        self.n_shared = int(n_shared)
        self.examples = tuple(examples)
        self.counts = dict(counts or {})


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProtocolSpec:
    """A registered split protocol.

    Parameters
    ----------
    name : str
        Public name used by :func:`make_folds` and the CLI.
    builder : callable
        ``builder(ws, seed=0, **kw) -> list[Fold]``.
    level : str
        Declared leakage level (see module docstring).
    doc : str
        One-line description for ``list_protocols`` / the CLI.
    params : dict
        Keyword parameters of the builder and their defaults (introspected).
    uses_seed : bool
        Whether the split depends on ``seed`` (deterministic protocols ignore it).
    """

    name: str
    builder: Callable[..., list[Fold]]
    level: str
    doc: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    uses_seed: bool = False

    def gap_for(self, kw: dict[str, Any]) -> float:
        """Guard gap (seconds) in force for a call with keyword arguments ``kw``."""
        if self.level != "time":
            return 0.0
        if "gap_s" in self.params:
            return float(kw.get("gap_s", self.params["gap_s"]))
        return 0.0

    def __call__(self, ws: "WindowSet", seed: int = 0, **kw: Any) -> list[Fold]:
        unknown = set(kw) - set(self.params)
        if unknown:
            raise TypeError(f"protocol {self.name!r} got unexpected keyword(s) {sorted(unknown)}; "
                            f"accepted: {sorted(self.params)}")
        return self.builder(ws, seed=seed, **kw)


PROTOCOLS: dict[str, ProtocolSpec] = {}
"""Registered protocols, in registration order (spec protocols first)."""

LEGACY_ALIASES: dict[str, str] = {
    # run_leakage_ladder.mendeley_folds
    "window": "window_random",
    "record": "record_random",
    "subject": "subject_loso",
    # run_leakage_ladder.bigideas_folds
    "reading": "record_kfold",
    "chrono0": "chronological_nogap",
    "chrono": "chronological",
    # run_bigideas_cv.py --protocol / run_subject_cv.py --folds
    "within": "chronological",
    "loso": "subject_loso",
}
"""Legacy script protocol names -> ppgbench protocol names."""


def register_protocol(name: str, level: str, doc: str = "", uses_seed: bool = False
                      ) -> Callable[[Callable[..., list[Fold]]], Callable[..., list[Fold]]]:
    """Decorator registering ``fn(ws, seed=0, **kw) -> list[Fold]`` under ``name``.

    The builder's keyword-only parameters (after ``ws`` and ``seed``) become the
    protocol's tunable parameters; their defaults are recorded so that
    :func:`leak_report` knows, for instance, the guard gap in force.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    def deco(fn: Callable[..., list[Fold]]) -> Callable[..., list[Fold]]:
        sig = inspect.signature(fn)
        params = {p.name: p.default for p in list(sig.parameters.values())[2:]
                  if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}
        if name in PROTOCOLS:
            log.warning("protocol %r re-registered", name)
        PROTOCOLS[name] = ProtocolSpec(name, fn, level, doc or (fn.__doc__ or "").strip().splitlines()[0],
                                       params, uses_seed)
        return fn

    return deco


def resolve_protocol(protocol: str) -> ProtocolSpec:
    """Return the :class:`ProtocolSpec` for a protocol name or legacy alias."""
    name = LEGACY_ALIASES.get(protocol, protocol)
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"unknown protocol {protocol!r}; known: {sorted(PROTOCOLS)} "
                         f"(aliases: {sorted(LEGACY_ALIASES)})") from None


def list_protocols() -> dict[str, str]:
    """``{name: "<level> -- <doc>"}`` for every registered protocol."""
    return {n: f"{s.level} -- {s.doc}" for n, s in PROTOCOLS.items()}


def declared_level(protocol: str) -> str:
    """Leakage level a protocol guarantees (``window``/``record``/``subject``/``time``).

    Accepts a parameter-variant key such as ``chronological_gap0`` (see
    :func:`protocol_variant`): the level is that of the base protocol.
    """
    return resolve_protocol(split_protocol_variant(protocol)[0]).level


# --------------------------------------------------------------------------- #
# Protocol variants: non-default parameters become part of the result key
# --------------------------------------------------------------------------- #
#: parameter -> spelling of a non-default value in a variant suffix (legacy ``run_bigideas_cv`` tags first)
_VARIANT_SPELLING: dict[str, Callable[[Any], str]] = {
    "gap_s": lambda v: f"gap{int(round(float(v)))}",                 # ``_gap0`` as in bigideas_within_1d_gap0
    "train_frac": lambda v: f"cut{int(round(float(v) * 100))}",      # ``_cut50`` as in bigideas_within_1d_cut50
    "n_splits": lambda v: f"k{int(v)}",
    "min_train": lambda v: f"mintrain{int(v)}",
    "min_test": lambda v: f"mintest{int(v)}",
}
#: one or more ``_<letters><digits>[p<digits>]`` groups at the end of a key (no base protocol ends like this)
_VARIANT_RE = re.compile(r"(?:_[a-z]+\d+(?:p\d+)?)+$")


def _variant_value(v: Any) -> str:
    """Digits-only spelling of a parameter value (``0.5`` -> ``0p5``, ``-3`` -> ``m3``)."""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        s = str(int(v))
    elif isinstance(v, (float, np.floating)):
        f = float(v)
        s = str(int(f)) if f.is_integer() else repr(f).replace(".", "p")
    else:
        s = re.sub(r"[^0-9.]", "", str(v)).replace(".", "p") or "0"
    return s.replace("-", "m")


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return bool(a == b)


def protocol_variant(protocol: str, kw: Mapping[str, Any] | None = None) -> str:
    """Key suffix encoding the *non-default* parameters of a protocol call (``""`` for defaults).

    A split built with a weaker guard gap, another chronological cut or a
    different number of folds is a different experiment.  Storing it under
    the canonical ``<dataset>/<protocol>/<model>/seed<k>`` key would silently
    replace the honest result and inherit its LaTeX macro name, so the CLI
    stores such runs under ``<protocol><variant>``.  The suffix follows the
    legacy ``run_bigideas_cv.py`` tags (``_gap0``, ``_cut50``) and extends
    them (``_k2`` for ``n_splits``, ``_mintrain32`` ...); parameters appear in
    the builder's signature order, e.g. ``chronological`` with
    ``train_frac=0.5, gap_s=0`` -> ``"_cut50_gap0"``.  Parameters equal to
    their default, and keywords the protocol does not take (``check``),
    contribute nothing.  Legacy aliases are accepted for ``protocol``.
    """
    spec = resolve_protocol(protocol)
    kw = dict(kw or {})
    parts: list[str] = []
    for name, default in spec.params.items():
        if name not in kw or _same_value(kw[name], default):
            continue
        spell = _VARIANT_SPELLING.get(name)
        parts.append(spell(kw[name]) if spell else re.sub(r"[^a-z]", "", name.lower()) + _variant_value(kw[name]))
    return "".join("_" + p for p in parts)


def split_protocol_variant(name: str) -> tuple[str, str]:
    """Inverse of :func:`protocol_variant`: ``"chronological_cut50_gap0"`` -> ``("chronological", "_cut50_gap0")``.

    A registered protocol name or alias is returned unchanged with an empty
    suffix; an unknown name without a recognisable suffix is returned as is
    (``resolve_protocol`` then raises the usual ``ValueError``).
    """
    if name in PROTOCOLS or name in LEGACY_ALIASES:
        return name, ""
    m = _VARIANT_RE.search(name)
    if m and m.start() > 0:
        base = name[:m.start()]
        if base in PROTOCOLS or base in LEGACY_ALIASES:
            return base, m.group(0)
    return name, ""


# --------------------------------------------------------------------------- #
# Time helper
# --------------------------------------------------------------------------- #
def window_time(ws: "WindowSet") -> np.ndarray | None:
    """Per-window reference time (POSIX s) or ``None`` when the dataset has none.

    Uses ``ws.time`` when present; otherwise falls back to the legacy
    ``meta['egv_time']`` (or ``meta['time']``) column mapped through
    ``ws.record`` exactly as ``run_bigideas_cv.chronological_split`` did.
    """
    t = getattr(ws, "time", None)
    if t is not None:
        return np.asarray(t, dtype=np.float64)
    meta = getattr(ws, "meta", None)
    if meta is not None and len(meta) and "record" in meta.columns:
        for col in ("egv_time", "time"):
            if col in meta.columns:
                return meta.set_index("record")[col].loc[np.asarray(ws.record)].to_numpy(np.float64)
    return None


def _require_time(ws: "WindowSet", what: str) -> np.ndarray:
    t = window_time(ws)
    if t is None:
        raise ValueError(f"{what} needs per-window reference times, but this WindowSet "
                         f"({getattr(ws, 'source', '') or 'unnamed'}) has `time=None` and no "
                         "`egv_time` column in `meta`")
    if not np.isfinite(t).all():
        raise ValueError(f"{what}: {int((~np.isfinite(t)).sum())} window time(s) are NaN/inf")
    return t


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _n(ws: "WindowSet") -> int:
    return int(len(ws))


def _as_folds(folds: Iterable[tuple[Any, Any]]) -> list[Fold]:
    return [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in folds]


@register_protocol("window_random", "window", "KFold(n_splits, shuffle) over windows -- leakiest rung", uses_seed=True)
def window_random(ws: "WindowSet", seed: int = 0, *, n_splits: int = 5) -> list[Fold]:
    """Shuffled KFold over windows (= ``run_leakage_ladder.mendeley_folds('window')``).

    Overlapping windows of the same recording land on both sides, so a model
    only has to memorise the recording to score well.
    """
    return _as_folds(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(_n(ws))))


@register_protocol("record_random", "record", "n_splits folds of shuffled records (subjects shared)", uses_seed=True)
def record_random(ws: "WindowSet", seed: int = 0, *, n_splits: int = 5) -> list[Fold]:
    """Records shuffled with ``RandomState(seed)`` and dealt round-robin into folds.

    Reproduces ``run_leakage_ladder.mendeley_folds('record')``: fold of a
    record = its position in the shuffled record list modulo ``n_splits``.
    """
    rec = np.asarray(ws.record)
    uniq, inv = np.unique(rec, return_inverse=True)
    recs = uniq.copy()
    np.random.RandomState(seed).shuffle(recs)                       # identical RNG use to the legacy code
    grp_of_uniq = np.empty(len(uniq), dtype=np.int64)
    grp_of_uniq[np.searchsorted(uniq, recs)] = np.arange(len(recs)) % n_splits
    g = grp_of_uniq[inv]
    return [(np.flatnonzero(g != k).astype(np.int64), np.flatnonzero(g == k).astype(np.int64))
            for k in range(n_splits)]


@register_protocol("record_kfold", "record", "GroupKFold(n_splits) by record (subjects shared)")
def record_kfold(ws: "WindowSet", seed: int = 0, *, n_splits: int = 5) -> list[Fold]:
    """Deterministic GroupKFold by record (= ``run_leakage_ladder.bigideas_folds('reading')``).

    All windows of one CGM reading stay together, but the neighbouring
    readings of the same subject (5 min apart) are in training.
    """
    return _as_folds(GroupKFold(n_splits).split(np.arange(_n(ws)), None, np.asarray(ws.record)))


@register_protocol("subject_kfold", "subject", "GroupKFold(n_splits) by subject")
def subject_kfold(ws: "WindowSet", seed: int = 0, *, n_splits: int = 5) -> list[Fold]:
    """Deterministic GroupKFold by subject (= ``run_subject_cv.py --folds k``)."""
    return _as_folds(GroupKFold(n_splits).split(np.arange(_n(ws)), None, np.asarray(ws.subject)))


@register_protocol("subject_loso", "subject", "leave-one-subject-out")
def subject_loso(ws: "WindowSet", seed: int = 0) -> list[Fold]:
    """Leave-one-subject-out, folds in sorted subject order (= ``run_subject_cv.py --folds loso``)."""
    return _as_folds(LeaveOneGroupOut().split(np.arange(_n(ws)), None, np.asarray(ws.subject)))


def chronological_split(ws: "WindowSet", train_frac: float = 0.7, gap_s: float = 3600.0) -> Fold:
    """Per-subject chronological train/test split of the windows (one fold).

    For every subject the cut is the ``train_frac`` quantile of that subject's
    *unique* reading times (so a reading with many windows counts once);
    training windows satisfy ``t < cut - gap_s`` and test windows ``t >= cut``.
    Identical to ``run_bigideas_cv.chronological_split``.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if gap_s < 0:
        raise ValueError(f"gap_s must be >= 0, got {gap_s}")
    t = _require_time(ws, "chronological_split")
    subject = np.asarray(ws.subject)
    tr, te = np.zeros(len(t), bool), np.zeros(len(t), bool)
    for s in np.unique(subject):
        m = subject == s
        cut = np.quantile(np.unique(t[m]), train_frac)
        tr |= m & (t < cut - gap_s)
        te |= m & (t >= cut)
    return np.flatnonzero(tr).astype(np.int64), np.flatnonzero(te).astype(np.int64)


@register_protocol("chronological", "time", "per subject first train_frac of readings -> train, guard gap gap_s")
def chronological(ws: "WindowSet", seed: int = 0, *, train_frac: float = 0.7, gap_s: float = 3600.0) -> list[Fold]:
    """One population fold: every subject's first ``train_frac`` of readings train,
    the rest test, with a ``gap_s`` guard gap (= ``bigideas_folds('chrono')``,
    ``run_bigideas_cv.py --protocol within``)."""
    return [chronological_split(ws, train_frac=train_frac, gap_s=gap_s)]


@register_protocol("chronological_nogap", "time", "chronological with gap_s = 0 (autocorrelation leak)")
def chronological_nogap(ws: "WindowSet", seed: int = 0, *, train_frac: float = 0.7) -> list[Fold]:
    """Chronological split without a guard gap (= ``bigideas_folds('chrono0')``).

    Honest at the record level but the last training reading and the first
    test reading are 5 min apart, so autocorrelation still leaks.
    """
    return [chronological_split(ws, train_frac=train_frac, gap_s=0.0)]


@register_protocol("personal", "time", "chronological, one fold per subject restricted to that subject")
def personal(ws: "WindowSet", seed: int = 0, *, train_frac: float = 0.7, gap_s: float = 3600.0,
             min_train: int = 64, min_test: int = 1) -> list[Fold]:
    """One fold per subject on that subject's own chronological split.

    Reproduces ``run_bigideas_cv.py --protocol personal``: a subject is kept
    only if it has *more than* ``min_train`` training windows (legacy
    ``len(tr) > 64``, so that a batch of 64 is always full) and at least
    ``min_test`` test windows.  Folds are in sorted subject order.
    """
    tr_all, te_all = chronological_split(ws, train_frac=train_frac, gap_s=gap_s)
    subject = np.asarray(ws.subject)
    folds: list[Fold] = []
    for s in np.unique(subject):
        tr, te = tr_all[subject[tr_all] == s], te_all[subject[te_all] == s]
        if len(tr) > min_train and len(te) >= min_test:
            folds.append((tr, te))
        else:
            log.info("personal: subject %s skipped (%d train / %d test windows)", s, len(tr), len(te))
    if not folds:
        raise ValueError(f"personal: no subject has > {min_train} training and >= {min_test} test windows")
    return folds


@register_protocol("chronological_blocked", "time",
                   "per subject n_splits contiguous time blocks; test one block, train outside +/- gap_s")
def chronological_blocked(ws: "WindowSet", seed: int = 0, *, n_splits: int = 5, gap_s: float = 3600.0) -> list[Fold]:
    """Blocked time-series cross-validation within subjects.

    Each subject's unique reading times are cut into ``n_splits`` contiguous
    blocks.  Fold ``k`` tests block ``k`` of every subject and trains on that
    subject's readings more than ``gap_s`` away from the block on *either*
    side.  Unlike ``chronological`` every reading is tested exactly once, so
    the within-person estimate does not depend on which 30 % happen to be
    last.  Subjects with fewer unique times than ``n_splits`` only contribute
    to the folds whose block is non-empty.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if gap_s < 0:
        raise ValueError(f"gap_s must be >= 0, got {gap_s}")
    t = _require_time(ws, "chronological_blocked")
    subject = np.asarray(ws.subject)
    tr = np.zeros((n_splits, len(t)), bool)
    te = np.zeros((n_splits, len(t)), bool)
    for s in np.unique(subject):
        m = subject == s
        ut = np.unique(t[m])
        for k, block in enumerate(np.array_split(ut, n_splits)):
            if not len(block):
                continue
            lo, hi = block[0], block[-1]
            te[k] |= m & (t >= lo) & (t <= hi)
            tr[k] |= m & ((t < lo - gap_s) | (t > hi + gap_s))
    folds = [(np.flatnonzero(tr[k]).astype(np.int64), np.flatnonzero(te[k]).astype(np.int64))
             for k in range(n_splits)]
    return [(a, b) for a, b in folds if len(a) and len(b)]


# --------------------------------------------------------------------------- #
# make_folds / verify
# --------------------------------------------------------------------------- #
def make_folds(ws: "WindowSet", protocol: str, seed: int = 0, **kw: Any) -> list[Fold]:
    """Build the folds of a named protocol and verify them at its declared level.

    Parameters
    ----------
    ws : WindowSet
    protocol : str
        A key of :data:`PROTOCOLS` or a legacy alias (:data:`LEGACY_ALIASES`).
    seed : int
        Shuffling seed; ignored by deterministic protocols.
    check : bool, keyword only, default True
        Run :func:`assert_no_leak` on the result at the protocol's declared
        level (with the guard gap in force).  Cheap -- integer set operations.
    **kw
        Protocol parameters, e.g. ``n_splits=5``, ``train_frac=0.7``,
        ``gap_s=3600``, ``min_train=64``.

    Returns
    -------
    list of ``(train_idx, test_idx)`` ``int64`` arrays.

    Raises
    ------
    ValueError   unknown protocol, or the dataset lacks what the protocol needs.
    TypeError    unknown protocol parameter.
    LeakError    the built folds violate the declared level (should never
                 happen for the shipped builders; a hard stop if it does).
    """
    check = bool(kw.pop("check", True))
    spec = resolve_protocol(protocol)
    folds = spec(ws, seed=seed, **kw)
    folds = [_validate_fold(tr, te, _n(ws), k, spec.name) for k, (tr, te) in enumerate(folds)]
    if check:
        assert_no_leak(ws, folds, spec.level, gap_s=spec.gap_for(kw), protocol=spec.name)
    log.debug("%s: %d fold(s), level %s, fingerprint %s", spec.name, len(folds), spec.level, fold_fingerprint(folds))
    return folds


def verify_folds(ws: "WindowSet", folds: Sequence[Fold], protocol: str, **kw: Any) -> str:
    """Assert ``folds`` at ``protocol``'s declared level; return that level.

    ``kw`` are the protocol parameters used to build the folds (only
    ``gap_s`` matters, for the ``time`` level).
    """
    spec = resolve_protocol(protocol)
    assert_no_leak(ws, folds, spec.level, gap_s=spec.gap_for(kw), protocol=spec.name)
    return spec.level


# --------------------------------------------------------------------------- #
# Leak detection
# --------------------------------------------------------------------------- #
def _validate_fold(tr: Any, te: Any, n: int, k: int, protocol: str | None = None) -> Fold:
    """Coerce one fold to ``int64`` index arrays (bool masks accepted) and reject malformed ones."""
    out = []
    for part, name in ((tr, "train"), (te, "test")):
        a = np.asarray(part)
        if a.dtype == bool:
            if a.shape != (n,):
                raise ValueError(f"fold {k}: boolean {name} mask has shape {a.shape}, expected ({n},)")
            a = np.flatnonzero(a)
        if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"fold {k}: {name} indices must be a 1-D integer array, got {a.dtype} {a.shape}")
        a = a.astype(np.int64, copy=False)
        if len(a) == 0:
            raise ValueError(f"fold {k}{_p(protocol)}: empty {name} set")
        if a.min() < 0 or a.max() >= n:
            raise ValueError(f"fold {k}{_p(protocol)}: {name} index out of range [0, {n}): "
                             f"min {a.min()}, max {a.max()}")
        n_dup = int(len(a) - np.count_nonzero(np.bincount(a, minlength=n)))
        if n_dup:
            raise ValueError(f"fold {k}{_p(protocol)}: {name} indices contain duplicates ({n_dup} repeated)")
        out.append(a)
    return out[0], out[1]


def _p(protocol: str | None) -> str:
    return f" of protocol {protocol!r}" if protocol else ""


class _Keys:
    """Integer codes of record / subject plus the time axis, computed once per WindowSet."""

    def __init__(self, ws: "WindowSet") -> None:
        self.n = _n(ws)
        rec, sub = np.asarray(ws.record), np.asarray(ws.subject)
        if rec.shape != (self.n,) or sub.shape != (self.n,):
            raise ValueError(f"ws.record {rec.shape} and ws.subject {sub.shape} must both be ({self.n},)")
        self.rec_uniq, self.rec = np.unique(rec, return_inverse=True)
        self.sub_uniq, self.sub = np.unique(sub, return_inverse=True)
        self.time: np.ndarray | None = window_time(ws)
        if self.time is not None and self.time.shape != (self.n,):
            raise ValueError(f"window time has shape {self.time.shape}, expected ({self.n},)")


def _presence(codes: np.ndarray, size: int) -> np.ndarray:
    """Boolean membership mask of integer codes (O(N), no sort or hash)."""
    m = np.zeros(size, bool)
    m[codes] = True
    return m


def _fold_counts(keys: _Keys, tr: np.ndarray, te: np.ndarray, gap_s: float, need_time: bool) -> dict[str, Any]:
    """Exact per-fold leak counts via presence masks over integer codes.

    Windows, records and subjects are all small integer codes, so every set
    operation is a boolean mask (linear time) rather than a sort / hash.
    """
    rec_tr, rec_te = keys.rec[tr], keys.rec[te]
    sub_tr, sub_te = keys.sub[tr], keys.sub[te]
    w_in_tr = _presence(tr, keys.n)
    r_in_tr, r_in_te = _presence(rec_tr, len(keys.rec_uniq)), _presence(rec_te, len(keys.rec_uniq))
    s_in_tr, s_in_te = _presence(sub_tr, len(keys.sub_uniq)), _presence(sub_te, len(keys.sub_uniq))
    shared_w = te[w_in_tr[te]]
    shared_r = np.flatnonzero(r_in_tr & r_in_te)
    shared_s = np.flatnonzero(s_in_tr & s_in_te)
    c: dict[str, Any] = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_records": int(r_in_tr.sum()), "test_records": int(r_in_te.sum()),
        "train_subjects": int(s_in_tr.sum()), "test_subjects": int(s_in_te.sum()),
        "shared_windows": int(len(shared_w)),
        "shared_records": int(len(shared_r)),
        "shared_record_test_windows": int(r_in_tr[rec_te].sum()),
        "shared_subjects": int(len(shared_s)),
        "shared_subject_test_windows": int(s_in_tr[sub_te].sum()),
        "_ex_windows": shared_w[:_N_EXAMPLES].tolist(),
        "_ex_records": keys.rec_uniq[shared_r[:_N_EXAMPLES]].tolist(),
        "_ex_subjects": keys.sub_uniq[shared_s[:_N_EXAMPLES]].tolist(),
        "gap_s": float(gap_s),
        "time_overlap_windows": None, "time_overlap_records": None, "min_test_train_distance_s": None,
        "test_in_train_span_windows": None, "time_nonfinite_windows": None,
        "_ex_time": [],
    }
    t = keys.time
    if t is None:
        if need_time:
            raise ValueError("level 'time' needs per-window reference times (ws.time or meta['egv_time'])")
        return c
    nonfinite = ~np.isfinite(t)
    c["time_nonfinite_windows"] = int(nonfinite[tr].sum() + nonfinite[te].sum())   # a NaN time is never "far away"
    if need_time and nonfinite.any():
        raise ValueError(f"level 'time': {int(nonfinite.sum())} window time(s) are NaN/inf")
    if not len(shared_s):                                    # no subject on both sides: nothing to check
        c.update(time_overlap_windows=0, time_overlap_records=0, test_in_train_span_windows=0)
        return c
    t_tr, t_te = t[tr], t[te]
    dist = np.full(len(te), np.inf)                          # distance to the nearest training reading (same subject)
    nearest = np.full(len(te), np.nan)
    in_span = np.zeros(len(te), bool)
    for s in shared_s:
        ts = np.sort(t_tr[sub_tr == s])
        m = sub_te == s
        tt = t_te[m]
        pos = np.searchsorted(ts, tt)
        left_i = np.clip(pos - 1, 0, len(ts) - 1)
        right_i = np.clip(pos, 0, len(ts) - 1)
        d_left = np.where(pos > 0, tt - ts[left_i], np.inf)
        d_right = np.where(pos < len(ts), ts[right_i] - tt, np.inf)
        use_left = d_left <= d_right
        dist[m] = np.where(use_left, d_left, d_right)
        nearest[m] = np.where(use_left, ts[left_i], ts[right_i])
        in_span[m] = (tt >= ts[0]) & (tt <= ts[-1])
    seen = np.isfinite(dist)
    inside = dist <= gap_s                                   # inf for unseen subjects never triggers
    c["time_overlap_windows"] = int(inside.sum())
    c["time_overlap_records"] = int(_presence(rec_te[inside], len(keys.rec_uniq)).sum()) if inside.any() else 0
    c["min_test_train_distance_s"] = float(dist[seen].min()) if seen.any() else None
    c["test_in_train_span_windows"] = int(in_span.sum())
    if inside.any():
        bad = np.flatnonzero(inside)
        first = bad[np.sort(np.unique(rec_te[bad], return_index=True)[1])][:_N_EXAMPLES]   # one example per record
        c["_ex_time"] = [{"record": str(keys.rec_uniq[rec_te[i]]), "subject": str(keys.sub_uniq[sub_te[i]]),
                          "t": float(t_te[i]), "nearest_train_t": float(nearest[i]), "distance_s": float(dist[i])}
                         for i in first]
    return c


def _public(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if not k.startswith("_")}


def _first_violation(c: dict[str, Any], level: str) -> tuple[str, int, list[Any]] | None:
    """Return ``(check, n_shared, examples)`` for the first failing check of ``level``."""
    for check in _CHECKS_FOR_LEVEL[level]:
        if check == "window" and c["shared_windows"]:
            return check, c["shared_windows"], c["_ex_windows"]
        if check == "record" and c["shared_records"]:
            return check, c["shared_records"], c["_ex_records"]
        if check == "subject" and c["shared_subjects"]:
            return check, c["shared_subjects"], c["_ex_subjects"]
        if check == "time" and c["time_overlap_windows"]:
            return check, c["time_overlap_windows"], c["_ex_time"]
    return None


def _message(k: int, n_folds: int, protocol: str | None, level: str, check: str, n: int,
             examples: list[Any], c: dict[str, Any]) -> str:
    head = f"fold {k + 1}/{n_folds}{_p(protocol)} violates level {level!r}: "
    if check == "window":
        body = f"{n} window index(es) appear in both train and test (e.g. {examples})"
    elif check == "record":
        body = (f"{n} record(s) appear in both train and test ({c['shared_record_test_windows']} test windows; "
                f"e.g. {examples})")
    elif check == "subject":
        body = (f"{n} subject(s) appear in both train and test ({c['shared_subject_test_windows']} test windows; "
                f"e.g. {examples})")
    else:
        ex = "; ".join(f"{e['record']} (subject {e['subject']}, t={e['t']:.0f}) is {e['distance_s']:.0f} s from "
                       f"the training reading at t={e['nearest_train_t']:.0f}" for e in examples)
        body = (f"{n} test window(s) of {c['time_overlap_records']} record(s) lie within gap_s={c['gap_s']:.0f} s "
                f"of a training reading of the same subject (e.g. {ex})")
    tail = (f". Train: {c['n_train']} windows / {c['train_records']} records / {c['train_subjects']} subjects; "
            f"test: {c['n_test']} / {c['test_records']} / {c['test_subjects']}. "
            f"Shared windows {c['shared_windows']}, records {c['shared_records']}, subjects {c['shared_subjects']}")
    if c["time_overlap_windows"] is not None:
        tail += f", time-overlap windows {c['time_overlap_windows']}"
    return head + body + tail + "."


def assert_no_leak(ws: "WindowSet", folds: Sequence[Fold], level: str, *, gap_s: float = 0.0,
                   protocol: str | None = None) -> None:
    """Raise :class:`LeakError` if any fold leaks at ``level``.

    Parameters
    ----------
    ws : WindowSet
    folds : sequence of ``(train_idx, test_idx)``
    level : {"window", "record", "subject", "time"}
        ``record`` also checks ``window``; ``subject`` also checks ``record``
        and ``window``; ``time`` also checks ``record`` and ``window`` and,
        for every subject present in both sets, that every test window's
        time is more than ``gap_s`` from the nearest training reading.
    gap_s : float
        Guard gap for the ``time`` level (ignored otherwise).
    protocol : str, optional
        Name used in the error message.

    Raises
    ------
    LeakError   precise message naming the fold, the violated check, the
                counts and up to five offending keys.
    ValueError  unknown level, malformed fold, or ``time`` requested without
                per-window times.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    keys = _Keys(ws)
    n_folds = len(folds)
    for k, (tr_raw, te_raw) in enumerate(folds):
        tr, te = _validate_fold(tr_raw, te_raw, keys.n, k, protocol)
        c = _fold_counts(keys, tr, te, gap_s, need_time=(level == "time"))
        hit = _first_violation(c, level)
        if hit is not None:
            check, n, ex = hit
            raise LeakError(_message(k, n_folds, protocol, level, check, n, ex, c), level=level, violated=check,
                            fold=k, protocol=protocol, n_shared=n, examples=ex, counts=_public(c))
    log.debug("assert_no_leak(level=%s, gap_s=%s): %d fold(s) clean", level, gap_s, n_folds)


def leak_report(ws: "WindowSet", folds: Sequence[Fold], protocol: str | None = None, *,
                gap_s: float | None = None) -> dict[str, Any]:
    """Exact per-fold and overall leak statistics (never raises on a leak).

    Parameters
    ----------
    ws : WindowSet
    folds : sequence of ``(train_idx, test_idx)``
    protocol : str, optional
        Protocol name or alias; fixes ``declared_level`` and, unless
        ``gap_s`` is given, the guard gap used for the time check.
    gap_s : float, optional
        Guard gap for the time statistics (default: the protocol's default,
        or 0).

    Returns
    -------
    dict with keys
        ``protocol``, ``declared_level``, ``gap_s``, ``n_windows``,
        ``n_records``, ``n_subjects``, ``n_folds``, ``has_time``,
        ``folds`` (list of per-fold dicts: ``n_train``, ``n_test``,
        ``train_records``, ``test_records``, ``train_subjects``,
        ``test_subjects``, ``shared_windows``, ``shared_records``,
        ``shared_record_test_windows``, ``shared_subjects``,
        ``shared_subject_test_windows``, ``time_overlap_windows``,
        ``time_overlap_records``, ``min_test_train_distance_s`` (distance to the
        nearest training reading of the same subject, over test windows whose
        subject is in train), ``test_in_train_span_windows`` (interpolation
        rather than extrapolation), ``time_nonfinite_windows`` -- the last
        five ``None`` without times),
        ``totals`` (sums of the shared counts and the overall minimum
        distance), ``test_coverage`` (fraction of
        windows in at least one test set), ``test_sets_disjoint``,
        ``time_finite`` (``None`` without times; ``False`` when any train/test
        window has a NaN/inf time, in which case ``passes["time"]`` is
        ``None`` rather than ``True``),
        ``passes`` (``{level: bool | None}``), ``declared_level_ok``,
        ``fingerprint``.
    """
    spec = resolve_protocol(protocol) if protocol else None
    if gap_s is None:
        gap_s = spec.gap_for({}) if spec else 0.0
    keys = _Keys(ws)
    rows: list[dict[str, Any]] = []
    tested = np.zeros(keys.n, np.int64)
    valid: list[Fold] = []
    for k, (tr_raw, te_raw) in enumerate(folds):
        tr, te = _validate_fold(tr_raw, te_raw, keys.n, k, spec.name if spec else protocol)
        valid.append((tr, te))
        c = _fold_counts(keys, tr, te, float(gap_s), need_time=False)
        np.add.at(tested, te, 1)
        rows.append({"fold": k, **_public(c)})
    has_time = keys.time is not None
    sums = {key: int(sum(r[key] for r in rows)) for key in
            ("shared_windows", "shared_records", "shared_record_test_windows", "shared_subjects",
             "shared_subject_test_windows")}
    if has_time:
        sums["time_overlap_windows"] = int(sum(r["time_overlap_windows"] or 0 for r in rows))
        sums["time_overlap_records"] = int(sum(r["time_overlap_records"] or 0 for r in rows))
        sums["test_in_train_span_windows"] = int(sum(r["test_in_train_span_windows"] or 0 for r in rows))
        sums["time_nonfinite_windows"] = int(sum(r["time_nonfinite_windows"] or 0 for r in rows))
        dists = [r["min_test_train_distance_s"] for r in rows if r["min_test_train_distance_s"] is not None]
        sums["min_test_train_distance_s"] = float(min(dists)) if dists else None
    else:
        sums.update(time_overlap_windows=None, time_overlap_records=None, min_test_train_distance_s=None,
                    test_in_train_span_windows=None, time_nonfinite_windows=None)
    time_finite = (sums["time_nonfinite_windows"] == 0) if has_time else None
    passes = {
        "window": sums["shared_windows"] == 0,
        "record": sums["shared_windows"] == 0 and sums["shared_records"] == 0,
        "subject": sums["shared_windows"] == 0 and sums["shared_records"] == 0 and sums["shared_subjects"] == 0,
        "time": (sums["shared_windows"] == 0 and sums["shared_records"] == 0 and sums["time_overlap_windows"] == 0)
                if time_finite else None,                         # NaN/inf times: cannot be verified, never "clean"
    }
    return {
        "protocol": spec.name if spec else protocol,
        "declared_level": spec.level if spec else None,
        "gap_s": float(gap_s),
        "n_windows": keys.n,
        "n_records": int(len(keys.rec_uniq)),
        "n_subjects": int(len(keys.sub_uniq)),
        "n_folds": len(rows),
        "has_time": has_time,
        "time_finite": time_finite,
        "folds": rows,
        "totals": sums,
        "test_coverage": float((tested > 0).mean()) if keys.n else 0.0,
        "test_sets_disjoint": bool((tested <= 1).all()),
        "passes": passes,
        "declared_level_ok": passes[spec.level] if spec else None,
        "fingerprint": fold_fingerprint(valid),
    }


# --------------------------------------------------------------------------- #
# Persistence / bookkeeping
# --------------------------------------------------------------------------- #
def fold_fingerprint(folds: Sequence[Fold]) -> str:
    """16-hex-digit SHA-256 of the fold indices: cite it next to a result so the
    exact split can be recognised later."""
    h = hashlib.sha256()
    for tr, te in folds:
        for a in (tr, te):
            a = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
            h.update(len(a).to_bytes(8, "little"))
            h.update(a.tobytes())
    return h.hexdigest()[:16]


def folds_to_json(folds: Sequence[Fold], path: str | None = None, **info: Any) -> str:
    """Serialise folds (plus free-form ``info`` such as protocol/seed/dataset) to JSON."""
    payload = {"info": info, "fingerprint": fold_fingerprint(folds),
               "folds": [{"train": np.asarray(tr, dtype=np.int64).tolist(),
                          "test": np.asarray(te, dtype=np.int64).tolist()} for tr, te in folds]}
    s = json.dumps(payload)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(s)
    return s


def folds_from_json(src: str) -> list[Fold]:
    """Inverse of :func:`folds_to_json`; ``src`` is a JSON string or a file path."""
    try:
        payload = json.loads(src)
    except (json.JSONDecodeError, ValueError):
        with open(src, encoding="utf-8") as fh:
            payload = json.load(fh)
    folds = [(np.asarray(f["train"], dtype=np.int64), np.asarray(f["test"], dtype=np.int64)) for f in payload["folds"]]
    fp = payload.get("fingerprint")
    if fp and fp != fold_fingerprint(folds):
        raise ValueError("fold fingerprint mismatch: the JSON was altered")
    return folds


def describe_folds(ws: "WindowSet", folds: Sequence[Fold], protocol: str | None = None) -> str:
    """Human-readable table: one line per fold with sizes and held-out subjects."""
    subject = np.asarray(ws.subject)
    lines = [f"{protocol or 'folds'}: {len(folds)} fold(s) over {_n(ws)} windows, fingerprint {fold_fingerprint(folds)}"]
    for k, (tr, te) in enumerate(folds):
        held = np.unique(subject[te])
        shown = ", ".join(map(str, held[:6])) + (" ..." if len(held) > 6 else "")
        lines.append(f"  fold {k + 1:2d}: train {len(tr):6d}  test {len(te):6d}  test subjects {len(held):3d} [{shown}]")
    return "\n".join(lines)
