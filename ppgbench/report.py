"""
ppgbench.report
===============
Result persistence and paper export: the part of the benchmark that makes
sure a manuscript never carries a hand-typed number.

Three concerns live here.

1. ``ResultStore`` -- one JSON file holding every (dataset, protocol, model,
   seed) result plus auxiliary entries (dataset statistics, bootstrap CIs).
   Writes are atomic (write-to-temp + ``os.replace``) so an interrupted run
   never leaves a half-written store behind, and record-level prediction CSVs
   are kept next to the store so that any statistic can be recomputed from
   the raw out-of-fold predictions.

2. ``to_latex_macros`` / ``to_markdown`` -- deterministic exporters.  The
   LaTeX export reproduces the naming of the legacy ``make_results_macros.py``
   exactly (``\\mendFusionMAE``, ``\\bigOneDWins``, ``\\mendLadderWindowMean``,
   ``\\mendFusionDiffCI`` ...) so the existing paper compiles unchanged
   against a store produced by the new framework.

3. Figure helpers (``fig_clarke``, ``fig_model_bars``, ``fig_tracking``,
   ``fig_ladder``) drawing with the validated paper palette of
   ``make_paper_figures.py``.

Key convention
--------------
Run entries are keyed ``"<dataset>/<protocol>/<model>/seed<k>"`` (see
``make_key`` / ``parse_key``).  Auxiliary entries use the reserved prefixes
``"dataset/<name>"`` (corpus statistics) and
``"bootstrap/<dataset>/<protocol>/<model>"`` (cluster-bootstrap CIs).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("ppgbench.report")

__all__ = [
    "PALETTE", "INK", "MUTED", "GRID", "NEUTRAL",
    "MODEL_NAMES", "MODEL_LABELS", "MODEL_ORDER", "METRIC_MACROS", "PROTOCOL_MACRO_NAMES",
    "DATASET_PREFIXES", "DEFAULT_PREFIX_MAP", "NULL_MODEL_BY_PROTOCOL",
    "make_key", "parse_key", "base_protocol", "macro_name", "fmt_pm", "latex_macro",
    "ResultStore", "to_latex_macros", "to_markdown", "aggregate",
    "paper_style", "save_figure", "fig_clarke", "fig_model_bars", "fig_tracking", "fig_ladder",
]

# --------------------------------------------------------------------------- #
# Palette and naming tables (shared with make_paper_figures.py / make_results_macros.py)
# --------------------------------------------------------------------------- #
#: categorical slots 1-4 of the validated palette: blue, orange, aqua, yellow
PALETTE: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e1", "#b3b2ad"

#: model name -> alphabetic CamelCase token used inside LaTeX macro names
MODEL_NAMES: dict[str, str] = {
    "mean": "Mean", "ridge": "Ridge", "demo": "Demo", "ridge+demo": "RidgeDemo",
    "1d": "OneD", "2d": "TwoD", "3d": "ThreeD", "fusion": "Fusion",
    "subjmean": "SubjMean", "subject_mean": "SubjMean",
    "gbm": "Gbm", "gbm+demo": "GbmDemo", "resnet1d": "ResNet", "cnnlstm": "CnnLstm",
    "tiny_transformer": "TinyTransformer", "transformer": "Transformer",
    "probe_identity": "ProbeIdentity", "probe_papagei": "ProbePapagei", "papagei": "Papagei",
    "anyppg": "Anyppg", "pulseppg": "Pulse", "tod": "Tod", "persist": "Persist",
}
#: model name -> human label for figure axes / legends
MODEL_LABELS: dict[str, str] = {
    "mean": "Train mean", "subject_mean": "Subject train mean", "subjmean": "Subject mean (leaky)",
    "ridge": "Ridge (PPG features)", "demo": "Ridge (demographics)", "ridge+demo": "Ridge (PPG + demo.)",
    "gbm": "GBM (PPG features)", "gbm+demo": "GBM (PPG + demo.)",
    "1d": "1-D CNN", "2d": "2-D CNN (CWT)", "3d": "PointNet (Takens)", "fusion": "Fusion (ours)",
    "resnet1d": "ResNet-1D", "cnnlstm": "CNN-LSTM", "tiny_transformer": "Tiny Transformer",
    "probe_identity": "Linear probe (features)", "probe_papagei": "PaPaGei probe",
}
#: canonical display order (null models first, classical, then networks, fusion last)
MODEL_ORDER: tuple[str, ...] = ("mean", "subject_mean", "demo", "ridge", "ridge+demo", "gbm", "gbm+demo",
                                "resnet1d", "cnnlstm", "tiny_transformer", "probe_identity", "probe_papagei",
                                "1d", "2d", "3d", "fusion")
#: macro suffix -> (metrics key, decimals), in legacy emission order
METRIC_MACROS: "OrderedDict[str, tuple[str, int]]" = OrderedDict([
    ("MAE", ("mae", 1)), ("RMSE", ("rmse", 1)), ("MARD", ("mard_pct", 1)),
    ("Rtwo", ("r2", 2)), ("R", ("pearson_r", 2)), ("ZoneA", ("zone_A_pct", 1)),
])
#: protocol -> CamelCase token for ladder macros (new names + legacy aliases)
PROTOCOL_MACRO_NAMES: dict[str, str] = {
    "window_random": "Window", "record_random": "Record", "record_kfold": "Reading", "subject_loso": "Subject",
    "subject_kfold": "SubjectKfold", "chronological_nogap": "ChronoNoGap", "chronological": "Chrono",
    "chronological_blocked": "ChronoBlocked", "personal": "Personal",
    # legacy run_leakage_ladder.py / run_bigideas_cv.py spellings
    "window": "Window", "record": "Record", "subject": "Subject", "reading": "Reading",
    "chrono0": "ChronoNoGap", "chrono": "Chrono", "within": "Chrono", "loso": "Subject",
}
#: per-dataset overrides of the protocol token (legacy called BIG IDEAs record folds "reading")
_PROTOCOL_MACRO_OVERRIDES: dict[str, dict[str, str]] = {"bigideas": {"record_random": "Reading"}}
#: dataset -> macro prefix
DATASET_PREFIXES: dict[str, str] = {"mendeley": "mend", "bigideas": "big"}
#: "<dataset>/<protocol>" -> experiment macro prefix (legacy: mend*, big*, bigLoso*)
DEFAULT_PREFIX_MAP: dict[str, str] = {
    "mendeley/subject_loso": "mend", "bigideas/chronological": "big", "bigideas/subject_loso": "bigLoso",
    "bigideas/personal": "bigPersonal",
    # legacy protocol spellings so a converted legacy store exports identically
    "mendeley/subject": "mend", "bigideas/within": "big", "bigideas/loso": "bigLoso",
}
#: the null model that per-subject "wins" and bootstrap differences are measured against
NULL_MODEL_BY_PROTOCOL: dict[str, str] = {
    "chronological": "subject_mean", "chronological_nogap": "subject_mean", "chronological_blocked": "subject_mean",
    "personal": "subject_mean", "within": "mean",
}
_DATASET_PREFIX, _BOOTSTRAP_PREFIX = "dataset/", "bootstrap/"
_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
                "7": "Seven", "8": "Eight", "9": "Nine"}


# --------------------------------------------------------------------------- #
# Keys, names and number formatting
# --------------------------------------------------------------------------- #
def make_key(dataset: str, protocol: str, model: str, seed: int) -> str:
    """Canonical store key ``"<dataset>/<protocol>/<model>/seed<k>"``."""
    for part in (dataset, protocol, model):
        if "/" in part:
            raise ValueError(f"'/' is reserved in store keys: {part!r}")
    return f"{dataset}/{protocol}/{model}/seed{int(seed)}"


def parse_key(key: str) -> dict[str, Any] | None:
    """Inverse of ``make_key``; ``None`` for auxiliary / foreign keys."""
    parts = key.split("/")
    if len(parts) != 4 or not re.fullmatch(r"seed\d+", parts[3]):
        return None
    return {"dataset": parts[0], "protocol": parts[1], "model": parts[2], "seed": int(parts[3][4:])}


#: one or more ``_<letters><digits>[p<digits>]`` groups at the end of a protocol key
_VARIANT_SUFFIX_RE = re.compile(r"(?:_[a-z]+\d+(?:p\d+)?)+$")


def base_protocol(protocol: str) -> str:
    """Strip a parameter-variant suffix: ``chronological_cut50_gap0`` -> ``chronological``.

    ``ppgbench.cli`` stores a run made with non-default protocol parameters
    under ``<protocol><variant>`` (the suffix produced by
    ``ppgbench.protocols.protocol_variant``: ``_gap<int>``, ``_cut<int>``,
    ``_k<int>`` ...) so that it can never overwrite the canonical result or
    borrow its macro name.  No base protocol name ends in ``_<letters><digits>``,
    so the suffix is unambiguous.  Used for every protocol-keyed lookup here
    (null model, macro token); a name without a suffix is returned unchanged.
    """
    m = _VARIANT_SUFFIX_RE.search(protocol)
    return protocol[:m.start()] if m and m.start() > 0 else protocol


def _camel(token: str) -> str:
    """Alphabetic CamelCase for an arbitrary identifier (digits spelled out)."""
    out = []
    for chunk in re.split(r"[^A-Za-z0-9]+", token):
        if not chunk:
            continue
        chunk = "".join(_DIGIT_WORDS[c] if c.isdigit() else c for c in chunk)
        out.append(chunk[0].upper() + chunk[1:])
    name = "".join(out)
    if not name:
        raise ValueError(f"cannot build an alphabetic macro token from {token!r}")
    return name


def model_token(model: str) -> str:
    """Macro token of a model name (``MODEL_NAMES`` first, CamelCase fallback)."""
    return MODEL_NAMES.get(model) or _camel(model)


def protocol_token(protocol: str, dataset: str | None = None) -> str:
    """Macro token of a protocol name (dataset-specific legacy aliases honoured).

    A parameter variant keeps the base token and appends the spelled-out
    suffix, so ``chronological_gap0`` -> ``ChronoGapZero`` never collides with
    the canonical ``Chrono`` macros.
    """
    base = base_protocol(protocol)
    variant = protocol[len(base):]
    if dataset and base in _PROTOCOL_MACRO_OVERRIDES.get(dataset, {}):
        tok = _PROTOCOL_MACRO_OVERRIDES[dataset][base]
    else:
        tok = PROTOCOL_MACRO_NAMES.get(base) or _camel(base)
    return tok + (_camel(variant) if variant else "")


def dataset_prefix(dataset: str) -> str:
    """Macro prefix of a dataset (``mend``, ``big``; lower-camel fallback)."""
    if dataset in DATASET_PREFIXES:
        return DATASET_PREFIXES[dataset]
    tok = _camel(dataset)
    return tok[0].lower() + tok[1:]


def macro_name(*parts: str) -> str:
    """Join tokens into a LaTeX-legal (purely alphabetic) macro name."""
    name = "".join(parts)
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"macro name must be alphabetic: {name!r}")
    return name


def fmt_pm(vals: Iterable[Any], nd: int) -> str:
    """``mean\\ensuremath{\\pm}std`` over seeds, a single value, or ``--``.

    Identical to ``make_results_macros._fmt``: ``None``/non-finite values are
    dropped first, population std (``np.std``) is used for the spread.
    """
    clean = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return "--"
    if len(clean) == 1:
        return f"{clean[0]:.{nd}f}"
    return f"{np.mean(clean):.{nd}f}\\ensuremath{{\\pm}}{np.std(clean):.{nd}f}"


def latex_macro(name: str, value: str) -> str:
    """One ``\\newcommand`` line."""
    return f"\\newcommand{{\\{macro_name(name)}}}{{{value}}}\n"


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and NaN into plain JSON values."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# ResultStore
# --------------------------------------------------------------------------- #
class ResultStore:
    """Insertion-ordered JSON store of benchmark results with atomic writes.

    Parameters
    ----------
    path : str | Path
        JSON file; loaded if it exists, created on first ``add``.
    autosave : bool
        Write after every ``add``/``remove`` (default).  Set ``False`` for
        batch updates and call ``save`` explicitly.

    Record-level predictions of a run are stored as CSV next to the store in
    ``<stem>_preds/<dataset>__<protocol>__<model>__seed<k>.csv`` with the
    columns ``record, subject, y, pred, n_windows`` (the legacy layout), so
    that bootstrap CIs, Clarke grids and tracking plots are always derived
    from the same out-of-fold predictions that produced the summary metrics.
    """

    def __init__(self, path: str | os.PathLike[str], autosave: bool = True) -> None:
        self.path = Path(path)
        self.autosave = autosave
        self._data: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        if self.path.exists():
            self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> "ResultStore":
        """(Re)load from disk; a missing file yields an empty store."""
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh, object_pairs_hook=OrderedDict)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{self.path}: expected a JSON object at top level")
            self._data = OrderedDict((str(k), dict(v)) for k, v in raw.items())
        else:
            self._data = OrderedDict()
        return self

    def save(self) -> Path:
        """Atomically write the store (temp file in the same directory + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(_jsonable(self._data), fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return self.path

    # -- mapping API --------------------------------------------------------
    def add(self, key: str, metrics: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
        """Insert or replace ``key`` with ``metrics`` (+ ``extra`` fields) and save."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        entry = _jsonable({**dict(metrics), **extra})
        parsed = parse_key(key)
        if parsed:                                   # make run entries self-describing
            for k, v in parsed.items():
                entry.setdefault(k, v)
        self._data[key] = entry
        if self.autosave:
            self.save()
        return entry

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._data.items())

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def remove(self, key: str) -> None:
        self._data.pop(key, None)
        if self.autosave:
            self.save()

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ResultStore({str(self.path)!r}, {len(self)} entries)"

    # -- queries ------------------------------------------------------------
    def runs(self, dataset: str | None = None, protocol: str | None = None,
             model: str | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Run entries as ``(parsed_key, metrics)`` pairs, optionally filtered."""
        out = []
        for key, m in self._data.items():
            p = parse_key(key)
            if p is None:
                continue
            if (dataset and p["dataset"] != dataset) or (protocol and p["protocol"] != protocol) \
                    or (model and p["model"] != model):
                continue
            out.append((p, m))
        return out

    def experiments(self) -> list[tuple[str, str]]:
        """Distinct ``(dataset, protocol)`` pairs in first-appearance order."""
        seen: "OrderedDict[tuple[str, str], None]" = OrderedDict()
        for p, _ in self.runs():
            seen.setdefault((p["dataset"], p["protocol"]), None)
        return list(seen)

    def groups(self, dataset: str, protocol: str) -> "OrderedDict[str, list[dict[str, Any]]]":
        """``model -> [metrics per seed, ascending seed]`` for one experiment."""
        g: "OrderedDict[str, list[tuple[int, dict[str, Any]]]]" = OrderedDict()
        for p, m in self.runs(dataset, protocol):
            g.setdefault(p["model"], []).append((p["seed"], m))
        return OrderedDict((k, [m for _, m in sorted(v, key=lambda t: t[0])]) for k, v in g.items())

    def to_dataframe(self) -> pd.DataFrame:
        """Flat table of run entries (one row per dataset/protocol/model/seed)."""
        rows = []
        for p, m in self.runs():
            row = {**p}
            row.update({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
            rows.append(row)
        return pd.DataFrame(rows)

    # -- record-level predictions ------------------------------------------
    @property
    def predictions_dir(self) -> Path:
        return self.path.with_name(f"{self.path.stem}_preds")

    def prediction_path(self, key: str) -> Path:
        return self.predictions_dir / (key.replace("/", "__") + ".csv")

    def write_predictions(self, key: str, df: pd.DataFrame) -> Path:
        """Write the record-level prediction table of ``key`` (atomic)."""
        required = {"record", "subject", "y", "pred"}
        if missing := required - set(df.columns):
            raise ValueError(f"prediction table lacks columns {sorted(missing)}")
        path = self.prediction_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return path

    def read_predictions(self, key: str) -> pd.DataFrame | None:
        """Prediction table of ``key`` or ``None`` if absent.  Ids are read as strings."""
        path = self.prediction_path(key)
        if not path.exists():
            return None
        return pd.read_csv(path, dtype={"record": str, "subject": str})

    def load_predictions(self, dataset: str, protocol: str, model: str) -> pd.DataFrame | None:
        """Seed-averaged record-level predictions of one model.

        Mirrors ``bootstrap_ci.load``: all seeds are concatenated and reduced to
        one row per (record, subject) with ``y`` taken once and ``pred`` averaged
        across seeds, which is the paired unit the cluster bootstrap resamples.
        """
        frames = [self.read_predictions(make_key(dataset, protocol, model, p["seed"]))
                  for p, _ in self.runs(dataset, protocol, model)]
        frames = [f for f in frames if f is not None]
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        agg = {"y": ("y", "first"), "pred": ("pred", "mean")}
        if "n_windows" in df.columns:
            agg["n_windows"] = ("n_windows", "first")
        if "time" in df.columns:
            agg["time"] = ("time", "first")
        return df.groupby(["record", "subject"], as_index=False).agg(**agg)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def aggregate(store: ResultStore, dataset: str, protocol: str, key: str = "mae") -> "OrderedDict[str, tuple[float, float, int]]":
    """``model -> (mean, std, n_seeds)`` of ``key`` over seeds (``make_paper_figures._aggregate``)."""
    out: "OrderedDict[str, tuple[float, float, int]]" = OrderedDict()
    for model, runs in store.groups(dataset, protocol).items():
        vals = [float(r[key]) for r in runs if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            out[model] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return out


def _lowest_seed(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return min(runs, key=lambda m: int(m.get("seed", 0)))


def _experiment_prefix(dataset: str, protocol: str, prefix_map: Mapping[str, str]) -> str:
    for k in (f"{dataset}/{protocol}", f"{dataset}/{protocol}/"):
        if k in prefix_map:
            return prefix_map[k]
    return dataset_prefix(dataset) + protocol_token(protocol, dataset)


def _null_model(dataset: str, protocol: str, models: Iterable[str], null_models: Mapping[str, str] | None) -> str | None:
    models = list(models)
    base = base_protocol(protocol)
    if null_models:
        for k in (f"{dataset}/{protocol}", f"{dataset}/{base}", protocol, base, dataset):
            if k in null_models and null_models[k] in models:
                return null_models[k]
    wanted = NULL_MODEL_BY_PROTOCOL.get(base, "mean")
    if wanted in models:
        return wanted
    return "mean" if "mean" in models else None


def dataset_stats_from_predictions(df: pd.DataFrame) -> dict[str, Any]:
    """Corpus statistics from a record-level table (``make_results_macros.dataset_stats``)."""
    y = df["y"].to_numpy(dtype=float)
    return {"n_records": int(len(df)), "n_subjects": int(df["subject"].nunique()),
            "n_windows": int(df["n_windows"].sum()) if "n_windows" in df.columns else None,
            "glucose_min": float(y.min()), "glucose_max": float(y.max()),
            "glucose_mean": float(y.mean()), "glucose_std": float(y.std())}


# --------------------------------------------------------------------------- #
# LaTeX export
# --------------------------------------------------------------------------- #
class _MacroSink:
    """Collects macros, refusing silent redefinition (LaTeX would error on it)."""

    def __init__(self) -> None:
        self.macros: "OrderedDict[str, str]" = OrderedDict()

    def add(self, name: str, value: str) -> None:
        name = macro_name(name)
        if name in self.macros and self.macros[name] != value:
            log.warning("macro \\%s emitted twice (%s vs %s); keeping the first", name, self.macros[name], value)
            return
        self.macros.setdefault(name, value)

    def render(self, header: str) -> str:
        return header + "".join(latex_macro(n, v) for n, v in self.macros.items())


def _experiment_macros(sink: _MacroSink, prefix: str, groups: Mapping[str, Sequence[dict[str, Any]]],
                       null_model: str | None) -> None:
    """Metric macros of one experiment, in exactly the legacy emission order."""
    for model, runs in groups.items():
        runs = [r for r in runs if r.get("mae") is not None]
        if not runs:
            continue
        tok = model_token(model)
        for mname, (key, nd) in METRIC_MACROS.items():
            sink.add(f"{prefix}{tok}{mname}", fmt_pm([r.get(key) for r in runs], nd))
        sink.add(f"{prefix}{tok}ZoneAB",
                 fmt_pm([r["zone_A_pct"] + r["zone_B_pct"] for r in runs
                         if r.get("zone_A_pct") is not None and r.get("zone_B_pct") is not None], 1))
        sink.add(f"{prefix}{tok}WithinR", fmt_pm([r.get("within_subject_r_median") for r in runs], 2))
        sink.add(f"{prefix}{tok}Seeds", str(len(runs)))
        sink.add(f"{prefix}{tok}N", str(runs[0].get("n", "--")))
        # extensions beyond the legacy set, emitted only when the metrics module provides them
        if any(r.get("parkes_zone_A_pct") is not None for r in runs):
            sink.add(f"{prefix}{tok}ParkesA", fmt_pm([r.get("parkes_zone_A_pct") for r in runs], 1))
            sink.add(f"{prefix}{tok}ParkesAB",
                     fmt_pm([r["parkes_zone_A_pct"] + r["parkes_zone_B_pct"] for r in runs
                             if r.get("parkes_zone_A_pct") is not None and r.get("parkes_zone_B_pct") is not None], 1))
        if any(r.get("bias") is not None for r in runs):
            sink.add(f"{prefix}{tok}Bias", fmt_pm([r.get("bias") for r in runs], 1))
    # per-subject wins against the null model (lowest seed of each)
    base = None
    if null_model and groups.get(null_model):
        base = _lowest_seed(groups[null_model]).get("per_subject")
    if base:
        for model, runs in groups.items():
            if model == null_model or not runs:
                continue
            ps = _lowest_seed(runs).get("per_subject")
            if not ps:
                continue
            wins = sum(ps[s]["mae"] < base[s]["mae"] for s in base
                       if s in ps and ps[s].get("mae") is not None and base[s].get("mae") is not None)
            sink.add(f"{prefix}{model_token(model)}Wins", str(wins))
            sink.add(f"{prefix}{model_token(model)}Worse", str(len(base) - wins))
    all_runs = [r for runs in groups.values() for r in runs]
    subjects = next((r["subjects"] for r in all_runs if r.get("subjects")), None)
    if subjects is not None:
        sink.add(f"{prefix}NSubjectsAll", str(len(subjects)))
    epochs = [int(r["epochs"]) for r in all_runs if r.get("epochs") is not None]
    if epochs:
        sink.add(f"{prefix}Epochs", str(max(epochs)))


def _dataset_macros(sink: _MacroSink, prefix: str, stats: Mapping[str, Any]) -> None:
    sink.add(f"{prefix}NRecords", str(stats["n_records"]))
    sink.add(f"{prefix}NSubjects", str(stats["n_subjects"]))
    if stats.get("n_windows") is not None:
        sink.add(f"{prefix}NWindows", str(int(stats["n_windows"])))
    sink.add(f"{prefix}GlucoseMin", f"{stats['glucose_min']:.0f}")
    sink.add(f"{prefix}GlucoseMax", f"{stats['glucose_max']:.0f}")
    sink.add(f"{prefix}GlucoseMean", f"{stats['glucose_mean']:.1f}")
    sink.add(f"{prefix}GlucoseStd", f"{stats['glucose_std']:.1f}")


def _bootstrap_macros(sink: _MacroSink, prefix: str, v: Mapping[str, Any]) -> None:
    sink.add(f"{prefix}Diff", f"{v['diff_vs_null']:+.1f}")
    sink.add(f"{prefix}DiffCI", f"[{v['diff_ci'][0]:+.1f}, {v['diff_ci'][1]:+.1f}]")
    sink.add(f"{prefix}MAECI", f"[{v['mae_ci'][0]:.1f}, {v['mae_ci'][1]:.1f}]")
    sink.add(f"{prefix}PBetter", f"{v['p_better_than_null']:.2f}")


def to_latex_macros(store: ResultStore, prefix_map: Mapping[str, str] | None = None,
                    null_models: Mapping[str, str] | None = None, ladder: bool = True,
                    header: str | None = None) -> str:
    """Render every result in ``store`` as ``\\newcommand`` macros.

    Naming reproduces ``make_results_macros.py``:

    * experiment metrics ``\\<prefix><Model><Metric>`` with ``Metric`` in
      ``MAE RMSE MARD Rtwo R ZoneA ZoneAB WithinR Seeds N``, multi-seed values
      as ``mean\\ensuremath{\\pm}std``;
    * ``\\<prefix><Model>ParkesA`` / ``ParkesAB`` / ``Bias`` when the metrics
      module reports the Parkes consensus grid and the signed bias (new);
    * ``\\<prefix><Model>Wins`` / ``Worse``: subjects on which the model beats
      the null model's per-subject MAE (chronological protocols);
    * ``\\<prefix>NSubjectsAll``, ``\\<prefix>Epochs``;
    * corpus statistics ``\\<ds>NRecords NSubjects NWindows GlucoseMin/Max/Mean/Std``
      from ``dataset/<name>`` entries (or, failing that, from the first run's
      prediction CSV);
    * bootstrap ``\\<prefix><Model>Diff DiffCI MAECI PBetter`` from
      ``bootstrap/<ds>/<proto>/<model>`` entries;
    * ladder ``\\<ds>Ladder<Protocol><Model>`` = MAE of every
      (dataset, protocol, model) group, so the leakage ladder needs no
      separate store.

    Parameters
    ----------
    prefix_map : ``"<dataset>/<protocol>" -> prefix``; defaults to
        ``DEFAULT_PREFIX_MAP`` (``mend``, ``big``, ``bigLoso``).  Unmapped
        experiments get ``<datasetPrefix><ProtocolToken>`` (e.g.
        ``syntheticSubjectLoso``), so nothing is silently dropped.
    null_models : ``"<dataset>/<protocol>" | protocol | dataset -> model`` used
        for Wins/Worse; default ``subject_mean`` for chronological protocols,
        else ``mean``.
    """
    prefix_map = {**DEFAULT_PREFIX_MAP, **(prefix_map or {})}
    sink = _MacroSink()
    # 1. experiments
    for dataset, protocol in store.experiments():
        groups = store.groups(dataset, protocol)
        prefix = _experiment_prefix(dataset, protocol, prefix_map)
        _experiment_macros(sink, prefix, groups, _null_model(dataset, protocol, groups, null_models))
    # 2. dataset statistics
    seen_ds: set[str] = set()
    for key, v in store.items():
        if key.startswith(_DATASET_PREFIX):
            ds = key[len(_DATASET_PREFIX):]
            _dataset_macros(sink, dataset_prefix(ds), v)
            seen_ds.add(ds)
    for dataset, protocol in store.experiments():
        if dataset in seen_ds:
            continue
        for p, _ in store.runs(dataset, protocol):
            df = store.read_predictions(make_key(dataset, protocol, p["model"], p["seed"]))
            if df is not None and len(df):
                _dataset_macros(sink, dataset_prefix(dataset), dataset_stats_from_predictions(df))
                seen_ds.add(dataset)
                break
    # 3. bootstrap CIs
    for key, v in store.items():
        if key.startswith(_BOOTSTRAP_PREFIX):
            parts = key[len(_BOOTSTRAP_PREFIX):].split("/")
            if len(parts) == 3:
                ds, proto, model = parts
                prefix = _experiment_prefix(ds, proto, prefix_map)
            elif len(parts) == 2:                           # legacy "<dataset>/<model>"
                ds, model = parts
                prefix = dataset_prefix(ds)
            else:
                log.warning("skipping malformed bootstrap key %s", key)
                continue
            _bootstrap_macros(sink, prefix + model_token(model), v)
    # 4. ladder
    if ladder:
        for dataset, protocol in store.experiments():
            for model, runs in store.groups(dataset, protocol).items():
                sink.add(f"{dataset_prefix(dataset)}Ladder{protocol_token(protocol, dataset)}{model_token(model)}",
                         fmt_pm([r.get("mae") for r in runs], 1))
    if header is None:
        header = f"% Auto-generated by ppgbench.report.to_latex_macros from {store.path.name} -- do not edit by hand.\n"
    return sink.render(header)


# --------------------------------------------------------------------------- #
# Markdown export
# --------------------------------------------------------------------------- #
def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


def _md_pm(vals: Iterable[Any], nd: int) -> str:
    return fmt_pm(vals, nd).replace("\\ensuremath{\\pm}", " ± ")


def to_markdown(store: ResultStore, title: str | None = None) -> str:
    """Human-readable summary: one table per experiment, bootstrap CIs, corpus statistics."""
    out = f"# {title or 'ppgbench results'}\n\nSource: `{store.path}`\n\n"
    for dataset, protocol in store.experiments():
        groups = store.groups(dataset, protocol)
        out += f"## {dataset} / {protocol}\n\n"
        rows = []
        order = [m for m in MODEL_ORDER if m in groups] + [m for m in groups if m not in MODEL_ORDER]
        for model in order:
            runs = [r for r in groups[model] if r.get("mae") is not None]
            if not runs:
                continue
            rows.append([
                MODEL_LABELS.get(model, model) + f" (`{model}`)", str(runs[0].get("n", "--")),
                _md_pm([r.get("mae") for r in runs], 1), _md_pm([r.get("rmse") for r in runs], 1),
                _md_pm([r.get("mard_pct") for r in runs], 1), _md_pm([r.get("r2") for r in runs], 2),
                _md_pm([r.get("pearson_r") for r in runs], 2),
                _md_pm([r.get("within_subject_r_median") for r in runs], 2),
                _md_pm([r.get("zone_A_pct") for r in runs], 1),
                _md_pm([r["zone_A_pct"] + r["zone_B_pct"] for r in runs
                        if r.get("zone_A_pct") is not None and r.get("zone_B_pct") is not None], 1),
                str(len(runs))])
        out += _md_table(["Model", "n", "MAE", "RMSE", "MARD %", "R²", "r", "within-r", "Zone A %", "A+B %", "seeds"],
                         rows) + "\n"
        boots = [(k, v) for k, v in store.items() if k.startswith(f"{_BOOTSTRAP_PREFIX}{dataset}/{protocol}/")]
        if boots:
            out += "Cluster bootstrap (subjects resampled) vs. null model:\n\n"
            out += _md_table(["Model", "MAE [95% CI]", "diff vs null [95% CI]", "P(better)", "subjects", "records"],
                             [[k.rsplit("/", 1)[1], f"{v['mae']:.2f} [{v['mae_ci'][0]:.2f}, {v['mae_ci'][1]:.2f}]",
                               f"{v['diff_vs_null']:+.2f} [{v['diff_ci'][0]:+.2f}, {v['diff_ci'][1]:+.2f}]",
                               f"{v['p_better_than_null']:.3f}", str(v.get("n_subjects", "--")),
                               str(v.get("n_records", "--"))] for k, v in boots]) + "\n"
    stats = [(k[len(_DATASET_PREFIX):], v) for k, v in store.items() if k.startswith(_DATASET_PREFIX)]
    if stats:
        out += "## Datasets\n\n" + _md_table(
            ["Dataset", "records", "subjects", "windows", "glucose min", "max", "mean", "std"],
            [[ds, str(v["n_records"]), str(v["n_subjects"]), str(v.get("n_windows", "--")),
              f"{v['glucose_min']:.0f}", f"{v['glucose_max']:.0f}", f"{v['glucose_mean']:.1f}",
              f"{v['glucose_std']:.1f}"] for ds, v in stats]) + "\n"
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
_RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
       "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
       "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
       "axes.labelcolor": INK, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42}


def paper_style() -> dict[str, Any]:
    """Apply the paper rcParams (``make_paper_figures.py``) and return them."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RC)
    return dict(_RC)


def save_figure(fig: Any, name: str, out_dir: str | os.PathLike[str], formats: Sequence[str] = ("pdf", "png"),
                dpi: int = 300) -> list[Path]:
    """Save ``fig`` as ``<out_dir>/<name>.<ext>`` for every format and close it."""
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    log.info("wrote %s", ", ".join(str(p) for p in paths))
    return paths


def _clarke_point(ref: float, pred: float) -> str:
    """Clarke (1987) zone of one pair; identical to ``plot_clarke_error_grid._classify_point``."""
    if (ref < 70 and abs(pred - ref) <= 20) or (ref >= 70 and abs(pred - ref) <= 0.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (ref >= 240 and pred <= 180) or (ref <= 70 and 70 <= pred < 180):
        return "D"
    if 70 <= ref <= 180 and (pred <= 70 or pred >= 240):
        return "D"
    if 70 < ref < 180 and abs(pred - ref) > 110:
        return "C"
    return "B"


def clarke_zone_pct(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """``{"zone_A_pct", ..., "zone_E_pct"}`` using ``ppgbench.metrics.clarke_zone`` when
    importable (the benchmark's single source of truth), else the local rule set."""
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    zones: Any = None
    try:
        from ppgbench.metrics import clarke_zone                    # lazy: parallel module
        zones = clarke_zone(y, pred)
    except Exception as exc:                                        # noqa: BLE001 - fallback by design
        log.debug("ppgbench.metrics.clarke_zone unavailable (%s); using local classifier", exc)
    if isinstance(zones, Mapping) and "zone_A_pct" in zones:
        return {f"zone_{z}_pct": float(zones[f"zone_{z}_pct"]) for z in "ABCDE"}
    if zones is None:
        zones = [_clarke_point(float(r), float(p)) for r, p in zip(y, pred)]
    labels = np.asarray([str(z).upper().strip() for z in np.asarray(zones).ravel()])
    n = max(1, len(labels))
    return {f"zone_{z}_pct": float((labels == z).sum() / n * 100) for z in "ABCDE"}


def _clarke_boundaries(ax: Any, hi: float) -> None:
    """Standard Clarke zone boundaries clipped to ``[0, hi]`` (``make_paper_figures``)."""
    segs = [([0, hi], [0, hi]), ([0, 175 / 3], [70, 70]), ([175 / 3, hi / 1.2], [70, hi]), ([70, 70], [84, hi]),
            ([0, 70], [180, 180]), ([70, 70 + (hi - 180)], [180, hi]), ([70, 70], [0, 56]),
            ([70, hi], [56, 0.8 * hi]), ([180, 180], [0, 70]), ([180, hi], [70, 70]), ([240, 240], [70, 180]),
            ([240, hi], [180, 180]), ([130, 180], [0, 70])]
    for i, (x, yy) in enumerate(segs):
        ax.plot(x, yy, color=MUTED, lw=0.6, ls=":" if i == 0 else "-", zorder=1)
    for z, (x, yy) in {"A": (40, 22), "B": (140, 230), "B ": (235, 150), "C": (100, 240), "C ": (165, 20),
                       "D": (30, 120), "D ": (245, 100), "E": (30, 220), "E ": (220, 30)}.items():
        if x < hi and yy < hi:
            ax.text(x, yy, z.strip(), color=MUTED, fontsize=8, ha="center", va="center", zorder=2)


def fig_clarke(ax: Any, y: np.ndarray, pred: np.ndarray, title: str | None = None,
               color: str = PALETTE[0]) -> dict[str, float]:
    """Clarke error grid of record-level predictions on ``ax``; returns zone percentages."""
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    hi = max(250.0, float(max(y.max(), pred.max())) + 10) if len(y) else 250.0
    _clarke_boundaries(ax, hi)
    ax.scatter(y, pred, s=9, c=color, alpha=0.75, linewidths=0.4, edgecolors="white", zorder=3)
    z = clarke_zone_pct(y, pred)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    ax.set_xlabel("Reference glucose (mg/dL)"); ax.set_ylabel("Predicted glucose (mg/dL)")
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.text(0.98, 0.02, "  ".join(f"{k}: {z[f'zone_{k}_pct']:.1f}%" for k in "ABCDE" if z[f"zone_{k}_pct"] > 0),
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7, color=INK,
            bbox={"fc": "white", "ec": "none", "pad": 1.5})
    return z


def fig_model_bars(ax: Any, agg: Mapping[str, tuple[float, float, int]], order: Sequence[str] | None = None,
                   labels: Mapping[str, str] | None = None, highlight: Sequence[str] = ("fusion",),
                   nets: Sequence[str] = ("1d", "2d", "3d"), xlabel: str = "Recording-level MAE (mg/dL)",
                   title: str | None = None) -> list[str]:
    """Horizontal MAE bars (mean ± std over seeds) per model; returns the plotted order.

    ``agg`` is ``aggregate(store, dataset, protocol)``: ``model -> (mean, std, n)``.
    Highlighted models are blue, single-view networks aqua, everything else neutral.
    """
    labels = {**MODEL_LABELS, **(labels or {})}
    order = list(order or MODEL_ORDER)
    models = [m for m in order if m in agg] + [m for m in agg if m not in order]
    if not models:
        raise ValueError("nothing to plot")
    mae = [agg[m][0] for m in models]; err = [agg[m][1] for m in models]
    cols = [PALETTE[0] if m in highlight else (PALETTE[2] if m in nets else NEUTRAL) for m in models]
    ax.barh(range(len(models)), mae, xerr=err, color=cols, height=0.62, error_kw={"lw": 0.8, "ecolor": INK})
    for i, v in enumerate(mae):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([labels.get(m, m) for m in models]); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_xlim(0, max(mae) * 1.18 if max(mae) > 0 else 1)
    if title:
        ax.set_title(title, loc="left", color=INK)
    return models


def fig_tracking(ax: Any, df_subject: pd.DataFrame, label: str = "Prediction from PPG",
                 reference_label: str = "CGM reference", spacing_min: float = 5.0,
                 title: str | None = None, color: str = PALETTE[0]) -> None:
    """Reference vs. prediction over the held-out period of ONE subject.

    Rows are ordered by ``time`` (POSIX s) when present, else by ``record``
    (the legacy BIG IDEAs record ids sort chronologically).  The x axis is in
    hours: elapsed time if ``time`` is present, else ``spacing_min`` per reading.
    """
    g = df_subject.copy()
    if "time" in g.columns and g["time"].notna().all():
        g = g.sort_values("time").reset_index(drop=True)
        hrs = (g["time"].to_numpy(dtype=float) - float(g["time"].iloc[0])) / 3600.0
    else:
        g = g.sort_values("record").reset_index(drop=True)
        hrs = np.arange(len(g)) * spacing_min / 60.0
    ax.plot(hrs, g["y"].to_numpy(dtype=float), color=INK, lw=1.2, label=reference_label)
    ax.plot(hrs, g["pred"].to_numpy(dtype=float), color=color, lw=1.2, label=label)
    sid = str(g["subject"].iloc[0]) if "subject" in g.columns and len(g) else ""
    ax.set_xlabel(f"Held-out time{', subject ' + sid if sid else ''} (h)"); ax.set_ylabel("Glucose (mg/dL)")
    ax.grid(axis="y"); ax.legend(frameon=False, loc="upper left", ncol=2)
    if title:
        ax.set_title(title, loc="left", color=INK)


def fig_ladder(ax: Any, store: ResultStore, dataset: str, protocols: Sequence[str],
               labels: Sequence[str] | None = None, models: Sequence[tuple[str, str, str]] | None = None,
               title: str | None = None) -> None:
    """Grouped MAE bars of several models across progressively less leaky protocols."""
    models = list(models or [("subject_mean", "Subject mean (leaky)", NEUTRAL), ("mean", "Null model", MUTED),
                             ("1d", "1-D CNN", PALETTE[0]), ("fusion", "Fusion", PALETTE[1])])
    aggs = {p: aggregate(store, dataset, p) for p in protocols}
    present = [m for m in models if any(m[0] in aggs[p] for p in protocols)]
    if not present:
        raise ValueError(f"no results for {dataset} under {list(protocols)}")
    w = 0.8 / len(present)
    for j, (m, lab, col) in enumerate(present):
        xs = np.arange(len(protocols)) + (j - (len(present) - 1) / 2) * w
        vals = [aggs[p][m][0] if m in aggs[p] else np.nan for p in protocols]
        errs = [aggs[p][m][1] if m in aggs[p] else 0.0 for p in protocols]
        ax.bar(xs, vals, w * 0.92, yerr=errs, color=col, label=lab, error_kw={"lw": 0.7, "ecolor": INK})
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(protocols))); ax.set_xticklabels(list(labels or protocols), fontsize=7)
    ax.set_ylabel("MAE (mg/dL)"); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    if title:
        ax.set_title(title, loc="left", color=INK)


def make_figures(store: ResultStore, out_dir: str | os.PathLike[str], dataset: str, protocol: str,
                 model: str = "fusion", formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Standard figure set for one experiment: Clarke grid of ``model``, model bars,
    and (for protocols with per-subject held-out periods) a tracking plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paper_style()
    written: list[Path] = []
    stem = f"{dataset}_{protocol}"
    agg = aggregate(store, dataset, protocol)
    if agg:
        fig, ax = plt.subplots(figsize=(3.6, 0.35 * len(agg) + 0.9))
        fig_model_bars(ax, agg, title=f"{dataset}, {protocol}")
        written += save_figure(fig, f"{stem}_models", out_dir, formats)
    if model not in agg:                                   # fall back to the best available network
        model = next((m for m in ("1d", "2d", "3d", "ridge", "mean") if m in agg), next(iter(agg), model))
    df = store.load_predictions(dataset, protocol, model)
    if df is not None and len(df) >= 2:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        fig_clarke(ax, df["y"].to_numpy(), df["pred"].to_numpy(),
                   f"{MODEL_LABELS.get(model, model)}, {len(df)} records, {df['subject'].nunique()} subjects")
        written += save_figure(fig, f"{stem}_clarke_{model}", out_dir, formats)
        if protocol.startswith("chrono") or protocol == "personal":
            sid = df["subject"].value_counts().index[0]
            fig, ax = plt.subplots(figsize=(4.4, 2.6))
            fig_tracking(ax, df[df["subject"] == sid], label=f"{MODEL_LABELS.get(model, model)} from PPG")
            written += save_figure(fig, f"{stem}_tracking_{model}", out_dir, formats)
    return written
