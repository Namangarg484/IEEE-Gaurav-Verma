"""
ppgbench.cli
============
Command-line front end of the benchmark (``python -m ppgbench ...``).

Subcommands
-----------
datasets   list registered datasets
models     list registered models and their kind (classical / net / probe)
protocols  list split protocols with the leak level each one guarantees
run        evaluate models under ONE protocol; writes the JSON store and the
           record-level prediction CSVs
ladder     the leakage ladder: the same models under several protocols
audit      leak report + integrity audit of a (dataset, protocol)
bootstrap  cluster-bootstrap CIs of every model against the null model
report     export a store to LaTeX macros / Markdown / figures
version    print the package version

Every sibling module (data, protocols, models, train, metrics, integrity)
is imported lazily inside the subcommand that needs it, so ``--help``,
``datasets`` and ``report`` keep working even when torch is slow to import
or a heavy dependency is missing.

Design rules (see docs/framework.md)
------------------------------------
* a result is only ever produced through ``make_folds`` + ``record_level`` +
  ``summarize``: there is no code path that reports window-level numbers;
* every run is keyed ``dataset/protocol/model/seed<k>`` in the store, and its
  out-of-fold record-level predictions are written next to it; a run made
  with non-default protocol parameters (``--protocol-arg gap_s=0``,
  ``train_frac=0.5``, ``n_splits=2`` ...) is keyed
  ``dataset/<protocol><variant>/...`` (``chronological_gap0``) so that it can
  never replace the canonical result or borrow its LaTeX macro name;
* folds are always verified at the protocol's declared level before a model
  sees them (``--protocol-arg check=False`` is ignored), and every stored
  run carries the fold fingerprint, the guard gap and the parameters in force;
* ``--quick`` (3 folds, 2 epochs) exists so the whole pipeline can be smoke
  tested in seconds; it is never a reportable configuration and is marked
  ``quick: true`` in the store.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ppgbench.report import (
    NULL_MODEL_BY_PROTOCOL, ResultStore, base_protocol, make_figures, make_key, to_latex_macros, to_markdown,
)

log = logging.getLogger("ppgbench.cli")

#: leak level each protocol must satisfy (the spec's table; ``protocols.declared_level`` is preferred when present)
PROTOCOL_LEVELS: dict[str, str] = {
    "window_random": "window", "record_random": "record", "record_kfold": "record",
    "subject_kfold": "subject", "subject_loso": "subject",
    "chronological": "time", "chronological_nogap": "time", "chronological_blocked": "time", "personal": "time",
}
#: default leakage-ladder rungs per dataset (legacy run_leakage_ladder.py: window/record/subject, reading/chrono0/chrono)
LADDER_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "mendeley": ("window_random", "record_random", "subject_loso"),
    "bigideas": ("record_kfold", "chronological_nogap", "chronological"),
}


def protocol_level(protocol: str) -> str:
    """Leak level a protocol guarantees: ``protocols.declared_level`` if the module provides it, else the spec table."""
    try:
        from ppgbench.protocols import declared_level
        return str(declared_level(protocol))
    except Exception:                                   # noqa: BLE001 - optional helper of a parallel module
        return PROTOCOL_LEVELS.get(base_protocol(protocol), "subject")
#: training budgets of the legacy scripts (fallback: Mendeley's)
DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "mendeley": {"epochs": 40, "batch": 16},
    "bigideas": {"epochs": 15, "batch": 64},
}
QUICK_FOLDS, QUICK_EPOCHS = 3, 2
DEFAULT_STORE = "results/ppgbench.json"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def parse_kv(items: Sequence[str] | None) -> dict[str, Any]:
    """``["k=v", ...]`` -> dict with Python literals parsed (``n=4`` -> 4, ``root=/x`` -> "/x")."""
    out: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def _device(name: str | None) -> Any:
    import torch
    if name:
        return torch.device(name)
    from ppgbench.train import device
    return device()


def _load_dataset(name: str, kwargs: Mapping[str, Any]) -> Any:
    from ppgbench.data.registry import load_dataset
    t0 = time.time()
    ws = load_dataset(name, **kwargs)
    log.info("loaded %s in %.1f s: %s", name, time.time() - t0, ws.describe())
    return ws


def _features(ws: Any) -> dict[str, Any]:
    """Classical-model inputs: hand-crafted PPG features and (if known) demographics."""
    from ppgbench.core.features import handcrafted_features
    return {"hand": handcrafted_features(ws.X, fs=ws.fs), "demo": ws.demographics(), "morph": None}


def _model_kinds(names: Sequence[str]) -> dict[str, str]:
    from ppgbench.models.registry import list_models
    kinds = list_models()
    unknown = [n for n in names if n not in kinds]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; available: {sorted(kinds)}")
    return {n: kinds[n] for n in names}


def _record_table(ws: Any) -> pd.DataFrame:
    """One row per record of a WindowSet (for corpus statistics)."""
    df = pd.DataFrame({"record": ws.record, "subject": ws.subject, "y": ws.y})
    return df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), n_windows=("y", "size"))


def dataset_stats(ws: Any) -> dict[str, Any]:
    """Corpus statistics at record level (what the paper's data table reports)."""
    rec = _record_table(ws)
    y = rec["y"].to_numpy(dtype=float)
    return {"n_records": int(len(rec)), "n_subjects": int(rec["subject"].nunique()), "n_windows": int(len(ws)),
            "glucose_min": float(y.min()), "glucose_max": float(y.max()),
            "glucose_mean": float(y.mean()), "glucose_std": float(y.std()),
            "fs": int(ws.fs), "window_samples": int(ws.X.shape[1]), "source": str(ws.source),
            "has_time": ws.time is not None}


def _result_line(key: str, m: Mapping[str, Any]) -> str:
    def f(k: str, fmt: str, suffix: str = "") -> str:
        v = m.get(k)
        return ("--" if v is None or not np.isfinite(v) else format(v, fmt)) + suffix
    return (f"RESULT {key:<44s} n={m.get('n', '--')!s:>5s}  MAE {f('mae', '6.2f')}  RMSE {f('rmse', '6.2f')}  "
            f"MARD {f('mard_pct', '5.1f', '%')}  R2 {f('r2', '+.3f')}  r {f('pearson_r', '+.3f')}  "
            f"zoneA {f('zone_A_pct', '5.1f', '%')}  ({m.get('seconds', 0):.0f} s)")


# --------------------------------------------------------------------------- #
# The evaluation loop shared by `run` and `ladder`
# --------------------------------------------------------------------------- #
class Experiment:
    """One dataset loaded once; views / features computed lazily and cached."""

    def __init__(self, dataset: str, dataset_kwargs: Mapping[str, Any], store: ResultStore,
                 device_name: str | None = None) -> None:
        self.dataset, self.store = dataset, store
        self.ws = _load_dataset(dataset, dataset_kwargs)
        self._views: dict[str, Any] | None = None
        self._feats: dict[str, Any] | None = None
        self._dev = None
        self._device_name = device_name
        store.add(f"dataset/{dataset}", dataset_stats(self.ws), kwargs=dict(dataset_kwargs))

    @property
    def views(self) -> dict[str, Any]:
        if self._views is None:
            from ppgbench.core.views import make_views
            t0 = time.time()
            self._views = make_views(self.ws.X)
            log.info("views computed in %.1f s: %s", time.time() - t0,
                     {k: tuple(v.shape) for k, v in self._views.items()})
        return self._views

    @property
    def feats(self) -> dict[str, Any]:
        if self._feats is None:
            self._feats = _features(self.ws)
        return self._feats

    @property
    def dev(self) -> Any:
        if self._dev is None:
            self._dev = _device(self._device_name)
        return self._dev

    # -- one fold ----------------------------------------------------------
    def predict_fold(self, name: str, kind: str, tr: np.ndarray, te: np.ndarray, seed: int,
                     epochs: int, batch: int, lr: float, model_kw: Mapping[str, Any]) -> np.ndarray:
        """Out-of-fold window predictions (mg/dL) of one model on one fold."""
        if kind == "net":
            from ppgbench.train import train_net
            if len(tr) < batch:                       # drop_last would yield zero batches
                batch_eff = max(2, len(tr) // 2)
                log.warning("train fold has %d windows < batch %d; using batch %d", len(tr), batch, batch_eff)
                batch = batch_eff
            return np.asarray(train_net(name, self.views, self.ws.y, tr, te, epochs, batch, lr, seed, self.dev,
                                        **model_kw), dtype=np.float32)
        from ppgbench.models.registry import build_model
        model = build_model(name, **model_kw)
        if hasattr(model, "fit") and hasattr(model, "predict"):
            return np.asarray(model.fit(self.ws, tr, self.feats).predict(self.ws, te, self.feats), dtype=np.float32)
        raise TypeError(f"model {name!r} (kind={kind}) has neither fit/predict nor is a registered net")

    # -- one (model, seed) -------------------------------------------------
    def run(self, protocol: str, name: str, kind: str, seed: int, epochs: int, batch: int, lr: float,
            quick: bool = False, fold_kwargs: Mapping[str, Any] | None = None,
            model_kw: Mapping[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        """Evaluate ``name`` under ``protocol`` for one seed; store metrics + predictions."""
        from ppgbench.metrics import record_level, summarize
        from ppgbench.protocols import fold_fingerprint, make_folds, protocol_variant, resolve_protocol
        ws = self.ws
        fold_kwargs = dict(fold_kwargs or {})
        if "check" in fold_kwargs:                      # the no-leak self-check is not optional for a stored result
            log.warning("ignoring protocol argument check=%r: a result is only stored for verified folds",
                        fold_kwargs.pop("check"))
        spec = resolve_protocol(protocol)
        folds = make_folds(ws, protocol, seed=seed, check=True, **fold_kwargs)   # LeakError aborts the run
        variant = protocol_variant(protocol, fold_kwargs)   # "" for default parameters; "_gap0", "_cut50_gap0" ...
        protocol_key = protocol + variant
        if variant:
            log.warning("non-default protocol parameters %s: storing under %r so the canonical %r result "
                        "is not overwritten", fold_kwargs, protocol_key, protocol)
        n_folds_total = len(folds)
        if quick:
            folds = folds[:QUICK_FOLDS]
        if not folds:
            raise SystemExit(f"protocol {protocol!r} produced no folds on {self.dataset}")
        pred = np.full(len(ws), np.nan, np.float32)
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            tr, te = np.asarray(tr), np.asarray(te)
            if len(te) == 0 or len(tr) == 0:
                log.warning("%s fold %d/%d is empty (train %d, test %d); skipped", name, k + 1, len(folds), len(tr), len(te))
                continue
            pred[te] = self.predict_fold(name, kind, tr, te, seed, epochs, batch, lr, model_kw or {})
            log.info("%-12s seed %d fold %2d/%d  n_train %5d n_test %5d  held-out %s  MAE(win) %.1f", name, seed,
                     k + 1, len(folds), len(tr), len(te), _short(np.unique(ws.subject[te])),
                     float(np.abs(pred[te] - ws.y[te]).mean()))
        keep = ~np.isnan(pred)
        if not keep.any():
            raise SystemExit(f"{name}: no out-of-fold predictions were produced")
        sub = ws.subset(keep)
        df = record_level(pred[keep], sub)
        if sub.time is not None and "time" not in df.columns:
            t = pd.DataFrame({"record": sub.record, "time": sub.time}).groupby("record")["time"].first()
            df["time"] = df["record"].map(t).to_numpy()
        m = dict(summarize(df))
        m["per_subject"] = {str(s): summarize(g) for s, g in df.groupby("subject") if len(g) > 5}
        m.update(dataset=self.dataset, protocol=protocol_key, base_protocol=protocol, variant=variant,
                 model=name, kind=kind, seed=int(seed),
                 folds=len(folds), n_folds_total=n_folds_total, quick=bool(quick),
                 subjects=sorted(np.unique(ws.subject).tolist()), n_windows_scored=int(keep.sum()),
                 seconds=round(time.time() - t0, 1), fold_kwargs=dict(fold_kwargs),
                 protocol_params={**spec.params, **fold_kwargs},
                 level=spec.level, gap_s=spec.gap_for(fold_kwargs) if spec.level == "time" else None,
                 fold_fingerprint=fold_fingerprint(folds), leak_checked=True)
        if kind == "net":
            m.update(epochs=int(epochs), batch=int(batch), lr=float(lr), device=str(self.dev))
        key = make_key(self.dataset, protocol_key, name, seed)
        self.store.write_predictions(key, df)
        self.store.add(key, m)
        return key, m


def _short(items: np.ndarray, n: int = 4) -> str:
    items = [str(i) for i in items]
    return ",".join(items[:n]) + (f",+{len(items) - n}" if len(items) > n else "")


def _seeds_for(kind: str, seeds: Sequence[int]) -> list[int]:
    """Nets get every seed; classical / probe models are deterministic -> seed 0 only (legacy)."""
    return list(seeds) if kind == "net" else [0]


def _run_models(exp: Experiment, protocols: Sequence[str], models: Sequence[str], args: argparse.Namespace) -> int:
    kinds = _model_kinds(models)
    defaults = DATASET_DEFAULTS.get(exp.dataset, DATASET_DEFAULTS["mendeley"])
    epochs = args.epochs if args.epochs is not None else (QUICK_EPOCHS if args.quick else defaults["epochs"])
    batch = args.batch if args.batch is not None else defaults["batch"]
    fold_kwargs = parse_kv(args.protocol_arg)
    model_kw = parse_kv(getattr(args, "model_arg", None))
    n = 0
    for protocol in protocols:
        for name in models:
            for seed in _seeds_for(kinds[name], args.seeds):
                key, m = exp.run(protocol, name, kinds[name], seed, epochs, batch, args.lr, args.quick,
                                 fold_kwargs, model_kw)
                print(_result_line(key, m), flush=True)
                n += 1
    log.info("%d runs written to %s (predictions in %s)", n, exp.store.path, exp.store.predictions_dir)
    return 0


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_datasets(args: argparse.Namespace) -> int:
    from ppgbench.data.registry import list_datasets
    for name in list_datasets():
        print(name)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from ppgbench.models.registry import list_models
    for name, kind in sorted(list_models().items(), key=lambda t: (t[1], t[0])):
        print(f"{name:<20s} {kind}")
    return 0


def cmd_protocols(args: argparse.Namespace) -> int:
    try:
        from ppgbench.protocols import list_protocols
        rows = list_protocols()
    except Exception:                                   # noqa: BLE001 - fall back to the spec table
        rows = {k: f"{v}" for k, v in PROTOCOL_LEVELS.items()}
    for name, desc in rows.items():
        print(f"{name:<22s} {desc}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    try:
        from ppgbench import __version__
    except Exception:                                   # noqa: BLE001 - __init__ belongs to another module
        __version__ = "unknown"
    print(f"ppgbench {__version__}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = ResultStore(args.out)
    exp = Experiment(args.dataset, parse_kv(args.dataset_arg), store, args.device)
    return _run_models(exp, [args.protocol], args.models, args)


def cmd_ladder(args: argparse.Namespace) -> int:
    store = ResultStore(args.out)
    protocols = args.protocols or list(LADDER_PROTOCOLS.get(args.dataset, LADDER_PROTOCOLS["mendeley"]))
    exp = Experiment(args.dataset, parse_kv(args.dataset_arg), store, args.device)
    return _run_models(exp, protocols, args.models, args)


def cmd_audit(args: argparse.Namespace) -> int:
    """Independent verification of a (dataset, protocol[, parameters]) split.

    The folds are built *without* the builder's own self-check and then
    verified here with the guard gap actually in force (the protocol default
    or ``--protocol-arg gap_s=...``), so a builder that ignored its gap is
    reported as a leak (exit code 1) instead of crashing, and the integrity
    report, the leak report and the assertion all use the same gap.
    """
    from ppgbench import integrity, protocols
    ws = _load_dataset(args.dataset, parse_kv(args.dataset_arg))
    fold_kwargs = parse_kv(args.protocol_arg)
    fold_kwargs.pop("check", None)
    spec = protocols.resolve_protocol(args.protocol)
    folds = protocols.make_folds(ws, args.protocol, seed=args.seed, check=False, **fold_kwargs)
    level = args.level or spec.level
    gap_s = spec.gap_for(fold_kwargs)                   # 0 unless the protocol is time-level
    report: dict[str, Any] = {"dataset": args.dataset, "protocol": args.protocol, "level": level, "gap_s": gap_s,
                              "fold_kwargs": fold_kwargs, "seed": int(args.seed),
                              "n_windows": int(len(ws)), "n_records": int(len(np.unique(ws.record))),
                              "n_subjects": int(len(np.unique(ws.subject))), "n_folds": len(folds),
                              "fingerprint": protocols.fold_fingerprint(folds)}
    try:
        report["audit"] = integrity.audit(ws, folds, args.protocol, level=level, gap_s=gap_s)
    except Exception as exc:                            # noqa: BLE001 - keep going, report the failure
        log.error("integrity.audit failed: %s", exc)
        report["audit"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        report["leak_report"] = protocols.leak_report(ws, folds, args.protocol, gap_s=gap_s)
    except Exception as exc:                            # noqa: BLE001
        report["leak_report"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        protocols.assert_no_leak(ws, folds, level, gap_s=gap_s, protocol=spec.name)
        report["no_leak"] = True
    except Exception as exc:                            # LeakError (or anything the check raises)
        report["no_leak"] = False
        report["leak_error"] = f"{type(exc).__name__}: {exc}"
    text = json.dumps(report, indent=2, default=str)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n", encoding="utf-8")
        log.info("wrote %s", args.json)
    print(text)
    gap_txt = f" gap_s={gap_s:g}" if level == "time" else ""
    print(f"AUDIT {args.dataset}/{args.protocol} level={level}{gap_txt}: "
          f"{'OK - no leak' if report['no_leak'] else 'LEAK DETECTED'}")
    return 0 if report["no_leak"] else 1


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from ppgbench.metrics import cluster_bootstrap
    store = ResultStore(args.store)
    groups = store.groups(args.dataset, args.protocol)
    if not groups:
        raise SystemExit(f"no runs for {args.dataset}/{args.protocol} in {args.store}")
    null = args.null or NULL_MODEL_BY_PROTOCOL.get(base_protocol(args.protocol), "mean")
    if null not in groups:
        raise SystemExit(f"null model {null!r} has no run under {args.dataset}/{args.protocol} "
                         f"(available: {list(groups)}); pass --null")
    df_null = store.load_predictions(args.dataset, args.protocol, null)
    if df_null is None:
        raise SystemExit(f"no prediction CSV for the null model {null!r}")
    models = args.models or [m for m in groups if m != null]
    print(f"{'model':<16s} {'MAE [95% CI]':>22s} {'diff vs ' + null + ' [95% CI]':>30s} {'P(better)':>10s}")
    for model in models:
        dfm = store.load_predictions(args.dataset, args.protocol, model)
        if dfm is None:
            log.warning("no predictions for %s; skipped", model)
            continue
        v = cluster_bootstrap(dfm, df_null, B=args.B, seed=args.seed)
        v = dict(v, null_model=null, B=int(args.B), seed=int(args.seed))
        store.add(f"bootstrap/{args.dataset}/{args.protocol}/{model}", v)
        if v.get("error"):
            print(f"{model:<16s} {v['error']}")
            continue
        print(f"{model:<16s} {v['mae']:6.2f} [{v['mae_ci'][0]:5.2f}, {v['mae_ci'][1]:5.2f}]   "
              f"{v['diff_vs_null']:+6.2f} [{v['diff_ci'][0]:+5.2f}, {v['diff_ci'][1]:+5.2f}]   "
              f"{v['p_better_than_null']:8.3f}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = ResultStore(args.store)
    if not len(store):
        raise SystemExit(f"{args.store} is empty or missing")
    prefix_map = parse_kv(args.prefix)
    wrote = False
    if args.latex:
        tex = to_latex_macros(store, prefix_map={k: str(v) for k, v in prefix_map.items()})
        Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.latex).write_text(tex, encoding="utf-8")
        log.info("wrote %s (%d macros)", args.latex, tex.count("\\newcommand"))
        wrote = True
    md = to_markdown(store)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(md, encoding="utf-8")
        log.info("wrote %s", args.markdown)
        wrote = True
    if args.figures:
        for dataset, protocol in store.experiments():
            for p in make_figures(store, args.figures, dataset, protocol, model=args.model):
                print("wrote", p)
        wrote = True
    if not wrote:
        print(md)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_dataset_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--dataset", required=required, default=None if required else "mendeley",
                   help="registered dataset name (see `datasets`)")
    p.add_argument("--dataset-arg", action="append", metavar="KEY=VALUE", default=[],
                   help="loader keyword, e.g. --dataset-arg n_subjects=4 (repeatable)")


def _add_protocol_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--protocol-arg", action="append", metavar="KEY=VALUE", default=[],
                   help="make_folds keyword, e.g. --protocol-arg n_splits=5 or gap_s=1800 (repeatable); "
                        "non-default values are stored under <protocol>_<variant>, e.g. chronological_gap1800")


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="seeds for network models")
    p.add_argument("--epochs", type=int, default=None, help="fixed epoch budget (dataset default; 2 with --quick)")
    p.add_argument("--batch", type=int, default=None, help="batch size (dataset default)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="torch device override (mps, cpu, cuda)")
    p.add_argument("--model-arg", action="append", metavar="KEY=VALUE", default=[],
                   help="build_model / train_net keyword (repeatable)")
    p.add_argument("--out", default=DEFAULT_STORE, help="JSON result store")
    p.add_argument("--quick", action="store_true", help=f"smoke test: first {QUICK_FOLDS} folds, {QUICK_EPOCHS} epochs")


def _add_verbosity(p: argparse.ArgumentParser, suppress: bool) -> None:
    d = argparse.SUPPRESS if suppress else None
    p.add_argument("-v", "--verbose", action="count", default=d if suppress else 0, help="-v info, -vv debug")
    p.add_argument("-q", "--quiet", action="store_true", default=d if suppress else False, help="warnings only")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ppgbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_verbosity(ap, suppress=False)
    common = argparse.ArgumentParser(add_help=False)      # so `-v`/`-q` also work after the subcommand
    _add_verbosity(common, suppress=True)
    subparsers = ap.add_subparsers(dest="command", required=True)

    class _Sub:                                            # add_parser with the shared parent
        @staticmethod
        def add_parser(name: str, **kw: Any) -> argparse.ArgumentParser:
            return subparsers.add_parser(name, parents=[common], **kw)
    sub = _Sub()

    sub.add_parser("datasets", help="list registered datasets").set_defaults(func=cmd_datasets)
    sub.add_parser("models", help="list registered models").set_defaults(func=cmd_models)
    sub.add_parser("protocols", help="list split protocols and their leak level").set_defaults(func=cmd_protocols)
    sub.add_parser("version", help="print the package version").set_defaults(func=cmd_version)

    p = sub.add_parser("run", help="evaluate models under one protocol")
    _add_dataset_args(p)
    p.add_argument("--protocol", default="subject_loso", help="split protocol (see docs/framework.md)")
    p.add_argument("--models", nargs="+", default=["mean", "ridge", "fusion"])
    _add_protocol_args(p); _add_train_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ladder", help="leakage ladder: same models under several protocols")
    _add_dataset_args(p)
    p.add_argument("--protocols", nargs="+", default=None, help="rungs (dataset default: see LADDER_PROTOCOLS)")
    p.add_argument("--models", nargs="+", default=["mean", "subject_mean", "1d"])
    _add_protocol_args(p); _add_train_args(p)
    p.set_defaults(func=cmd_ladder)

    p = sub.add_parser("audit", help="leak report + integrity audit")
    _add_dataset_args(p)
    p.add_argument("--protocol", default="subject_loso")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", choices=["window", "record", "subject", "time"], default=None,
                   help="leak level to assert (default: the protocol's declared level)")
    p.add_argument("--json", default=None, help="also write the report here")
    _add_protocol_args(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("bootstrap", help="cluster-bootstrap CIs vs. the null model")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--models", nargs="*", default=None, help="default: every model except the null")
    p.add_argument("--null", default=None, help="null model (default: subject_mean for chronological, else mean)")
    p.add_argument("-B", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("report", help="export LaTeX macros / Markdown / figures")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--latex", default=None, help="write \\newcommand macros here")
    p.add_argument("--markdown", default=None, help="write a Markdown summary here")
    p.add_argument("--figures", default=None, help="write the standard figures into this directory")
    p.add_argument("--model", default="fusion", help="model drawn in the Clarke / tracking figures")
    p.add_argument("--prefix", action="append", metavar="DATASET/PROTOCOL=PREFIX", default=[],
                   help="macro prefix override, e.g. --prefix mendeley/subject_loso=mend")
    p.set_defaults(func=cmd_report)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose > 1 else logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    return int(args.func(args) or 0)


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())
