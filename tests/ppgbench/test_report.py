"""
Tests for ppgbench.report: the result store, the LaTeX/Markdown exporters and
the figure helpers.

The LaTeX naming tests are *parity* tests: the same numbers are pushed through
the legacy ``make_results_macros.py`` (imported from the repository root) and
through ``to_latex_macros``; the macro sets must be identical.  They need no
dataset, no torch and run in well under a second.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import make_results_macros as legacy  # noqa: E402  (repository root; the reference behaviour)
from ppgbench import report  # noqa: E402
from ppgbench.report import (  # noqa: E402
    ResultStore, aggregate, fmt_pm, macro_name, make_key, model_token, parse_key, protocol_token,
    to_latex_macros, to_markdown,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _metrics(rng: np.random.Generator, n: int = 67, per_subject: bool = False, subjects: int = 23) -> dict:
    m = {"n": n, "mae": float(rng.uniform(10, 16)), "rmse": float(rng.uniform(15, 20)),
         "mard_pct": float(rng.uniform(9, 14)), "r2": float(rng.uniform(-0.3, 0.3)),
         "pearson_r": float(rng.uniform(-0.6, 0.6)),
         "within_subject_r_median": None if rng.uniform() < 0.3 else float(rng.uniform(-0.5, 0.5)),
         "zone_A_pct": float(rng.uniform(70, 90)), "zone_B_pct": float(rng.uniform(5, 10)),
         "zone_C_pct": 0.0, "zone_D_pct": float(rng.uniform(0, 2)), "zone_E_pct": 0.0, "epochs": 40}
    if per_subject:
        m["per_subject"] = {f"{s:03d}": {"n": 20, "mae": float(rng.uniform(8, 20))} for s in range(1, subjects + 1)}
        m["subjects"] = [f"{s:03d}" for s in range(1, subjects + 1)]
        m["epochs"] = 15
    return m


def _legacy_summary(rng: np.random.Generator, models: dict[str, int], prefix: str = "",
                    per_subject: bool = False) -> dict:
    """Legacy ``subject_cv_summary.json`` / ``bigideas_summary.json`` layout."""
    out = {}
    for model, n_seeds in models.items():
        for seed in range(n_seeds):
            m = _metrics(rng, per_subject=per_subject)
            m.update(model=model, seed=seed)
            out[f"{prefix}{model}_seed{seed}"] = m
    return out


def _store_from_legacy(tmp_path: Path, summary: dict, dataset: str, protocol: str, prefix: str = "") -> ResultStore:
    store = ResultStore(tmp_path / f"{dataset}_{protocol}.json", autosave=False)
    for tag, m in summary.items():
        model = tag[len(prefix):].rsplit("_seed", 1)[0]
        store.add(make_key(dataset, protocol, model, m["seed"]), m)
    store.save()
    return store


def _macro_dict(tex: str, prefix: str | None = None, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    out = {}
    for line in tex.splitlines():
        if not line.startswith("\\newcommand{\\"):
            continue
        name, value = line[len("\\newcommand{\\"):].split("}{", 1)
        value = value[:-1]
        if prefix and not name.startswith(prefix):
            continue
        if any(name.startswith(e) for e in exclude):
            continue
        out[name] = value
    return out


# --------------------------------------------------------------------------- #
# naming and formatting primitives
# --------------------------------------------------------------------------- #
def test_fmt_pm_matches_legacy():
    cases = [[1.2345], [1.0, 2.0, 4.0], [None, 3.3, float("nan")], [], [None]]
    for vals in cases:
        for nd in (1, 2):
            assert fmt_pm(vals, nd) == legacy._fmt(vals, nd)


def test_fmt_pm_uses_population_std_and_pm_macro():
    s = fmt_pm([1.0, 2.0, 3.0], 2)
    assert s == "2.00\\ensuremath{\\pm}0.82"          # np.std (ddof=0)


def test_model_tokens_cover_legacy_names():
    for name, token in legacy.NAMES.items():
        assert model_token(name) == token
    assert model_token("subject_mean") == "SubjMean"
    assert model_token("tiny_transformer") == "TinyTransformer"
    assert macro_name(model_token("probe-2x"))         # digits spelled out, alphabetic only
    assert model_token("probe-2x") == "ProbeTwox"


def test_protocol_tokens():
    assert protocol_token("window_random") == "Window"
    assert protocol_token("record_random") == "Record"
    assert protocol_token("record_random", "bigideas") == "Reading"   # legacy BIG IDEAs spelling
    assert protocol_token("chronological_nogap") == "ChronoNoGap"
    assert protocol_token("chronological") == "Chrono"
    assert protocol_token("subject_loso") == "Subject"
    assert protocol_token("some_new-protocol") == "SomeNewProtocol"


def test_macro_name_rejects_non_alpha():
    with pytest.raises(ValueError):
        macro_name("mend", "1d")


def test_key_roundtrip():
    key = make_key("mendeley", "subject_loso", "ridge+demo", 2)
    assert key == "mendeley/subject_loso/ridge+demo/seed2"
    assert parse_key(key) == {"dataset": "mendeley", "protocol": "subject_loso", "model": "ridge+demo", "seed": 2}
    assert parse_key("dataset/mendeley") is None
    assert parse_key("bootstrap/mendeley/subject_loso/fusion") is None
    with pytest.raises(ValueError):
        make_key("a/b", "p", "m", 0)


# --------------------------------------------------------------------------- #
# ResultStore
# --------------------------------------------------------------------------- #
def test_store_roundtrip_and_atomic_write(tmp_path):
    path = tmp_path / "sub" / "store.json"
    store = ResultStore(path)
    assert len(store) == 0 and not path.exists()
    entry = store.add(make_key("synthetic", "subject_loso", "mean", 0),
                      {"mae": np.float32(1.5), "n": np.int64(4), "nan": float("nan"), "arr": np.arange(2)},
                      seconds=0.5)
    assert path.exists() and not list(path.parent.glob(".store.json.*.tmp"))
    assert entry["dataset"] == "synthetic" and entry["model"] == "mean" and entry["seed"] == 0
    assert entry["nan"] is None and entry["arr"] == [0, 1]
    raw = json.loads(path.read_text())
    assert list(raw) == ["synthetic/subject_loso/mean/seed0"]
    assert isinstance(raw[list(raw)[0]]["mae"], float)
    again = ResultStore(path)
    assert again.get("synthetic/subject_loso/mean/seed0")["seconds"] == 0.5
    assert "synthetic/subject_loso/mean/seed0" in again and again.keys() == store.keys()
    again.remove("synthetic/subject_loso/mean/seed0")
    assert len(ResultStore(path)) == 0


def test_store_queries_and_order(tmp_path):
    store = ResultStore(tmp_path / "s.json", autosave=False)
    for seed in (2, 0, 1):
        store.add(make_key("d", "subject_loso", "1d", seed), {"mae": 10 + seed})
    store.add(make_key("d", "subject_loso", "mean", 0), {"mae": 14.0})
    store.add(make_key("d", "window_random", "mean", 0), {"mae": 9.0})
    store.add("dataset/d", {"n_records": 1})
    assert store.experiments() == [("d", "subject_loso"), ("d", "window_random")]
    g = store.groups("d", "subject_loso")
    assert list(g) == ["1d", "mean"] and [m["mae"] for m in g["1d"]] == [10, 11, 12]
    assert aggregate(store, "d", "subject_loso")["1d"] == (11.0, pytest.approx(np.std([10, 11, 12])), 3)
    assert len(store.runs(model="mean")) == 2
    df = store.to_dataframe()
    assert set(df.columns) >= {"dataset", "protocol", "model", "seed", "mae"} and len(df) == 5


def test_store_predictions_roundtrip_and_seed_average(tmp_path):
    store = ResultStore(tmp_path / "s.json", autosave=False)
    df0 = pd.DataFrame({"record": ["01_0001", "01_0002"], "subject": ["01", "01"], "y": [100.0, 120.0],
                        "pred": [110.0, 110.0], "n_windows": [3, 3]})
    df1 = df0.assign(pred=[90.0, 130.0])
    for seed, df in enumerate((df0, df1)):
        key = make_key("mendeley", "subject_loso", "fusion", seed)
        store.add(key, {"mae": 1.0})
        p = store.write_predictions(key, df)
        assert p.exists() and p.parent == store.predictions_dir
    back = store.read_predictions(make_key("mendeley", "subject_loso", "fusion", 0))
    assert back["subject"].tolist() == ["01", "01"] and back["record"].tolist()[0] == "01_0001"   # ids stay strings
    avg = store.load_predictions("mendeley", "subject_loso", "fusion")
    assert avg["pred"].tolist() == [100.0, 120.0] and avg["y"].tolist() == [100.0, 120.0]
    assert store.load_predictions("mendeley", "subject_loso", "nope") is None
    with pytest.raises(ValueError):
        store.write_predictions("x/y/z/seed0", pd.DataFrame({"record": []}))


# --------------------------------------------------------------------------- #
# LaTeX parity with make_results_macros.py
# --------------------------------------------------------------------------- #
def test_latex_experiment_macros_match_legacy_mendeley(tmp_path):
    rng = np.random.default_rng(0)
    summary = _legacy_summary(rng, {"mean": 1, "ridge": 1, "demo": 1, "ridge+demo": 1, "1d": 3, "2d": 3,
                                    "3d": 3, "fusion": 3, "gbm": 1, "resnet1d": 1, "cnnlstm": 1})
    store = _store_from_legacy(tmp_path, summary, "mendeley", "subject_loso")
    expected = _macro_dict(legacy.experiment(summary, "", "mend") + legacy._macro("mendEpochs", "40"))
    got = _macro_dict(to_latex_macros(store), prefix="mend", exclude=("mendLadder",))
    assert got == expected
    assert "mendFusionMAE" in got and "\\ensuremath{\\pm}" in got["mendFusionMAE"]


def test_latex_experiment_macros_match_legacy_bigideas_with_wins(tmp_path):
    rng = np.random.default_rng(1)
    within = _legacy_summary(rng, {"mean": 1, "ridge": 1, "1d": 2, "fusion": 1}, "within_", per_subject=True)
    loso = _legacy_summary(rng, {"mean": 1, "1d": 1}, "loso_")
    # legacy main(): experiment() blocks + Wins/Worse + NSubjectsAll + Epochs
    exp = legacy.experiment(within, "within_", "big") + legacy.experiment(loso, "loso_", "bigLoso")
    base = within["within_mean_seed0"]["per_subject"]
    for t, m in within.items():
        if t.endswith("_seed0") and t != "within_mean_seed0":
            model = t[len("within_"):-len("_seed0")]
            wins = sum(m["per_subject"][k]["mae"] < base[k]["mae"] for k in base)
            exp += legacy._macro(f"big{legacy.NAMES[model]}Wins", str(wins))
            exp += legacy._macro(f"big{legacy.NAMES[model]}Worse", str(len(base) - wins))
    exp += legacy._macro("bigNSubjectsAll", "23") + legacy._macro("bigEpochs", "15")
    exp += legacy._macro("bigLosoEpochs", "40")                # new: every experiment gets an Epochs macro
    store = ResultStore(tmp_path / "big.json", autosave=False)
    for tag, m in within.items():                               # legacy "mean" under within == subject_mean
        model = tag[len("within_"):].rsplit("_seed", 1)[0]
        store.add(make_key("bigideas", "chronological", "subject_mean" if model == "mean" else model, m["seed"]), m)
    for tag, m in loso.items():
        store.add(make_key("bigideas", "subject_loso", tag[len("loso_"):].rsplit("_seed", 1)[0], m["seed"]), m)
    got = _macro_dict(to_latex_macros(store), prefix="big", exclude=("bigLadder",))
    expected = _macro_dict(exp)
    # the legacy null under `within_` was called "mean"; the framework names it subject_mean -> SubjMean
    expected = {k.replace("bigMean", "bigSubjMean") if k.startswith("bigMean") else k: v for k, v in expected.items()}
    assert got == expected
    assert got["bigOneDWins"].isdigit() and int(got["bigOneDWins"]) + int(got["bigOneDWorse"]) == 23


def test_latex_ladder_macros_match_legacy(tmp_path):
    rng = np.random.default_rng(2)
    legacy_ladder, store = {}, ResultStore(tmp_path / "ladder.json", autosave=False)
    new_proto = {"mendeley": {"window": "window_random", "record": "record_random", "subject": "subject_loso"},
                 "bigideas": {"reading": "record_random", "chrono0": "chronological_nogap", "chrono": "chronological"}}
    for ds, protos in new_proto.items():
        for old, new in protos.items():
            for model, n_seeds in (("mean", 1), ("subjmean", 1), ("1d", 3), ("fusion", 2)):
                for seed in range(n_seeds):
                    m = _metrics(rng)
                    legacy_ladder[f"{ds}/{old}/{model}/seed{seed}"] = m
                    store.add(make_key(ds, new, "subject_mean" if model == "subjmean" else model, seed), m)
    # legacy main() ladder block, verbatim
    groups: dict[str, list[float]] = {}
    for key, v in legacy_ladder.items():
        ds, proto, model, _ = key.split("/")
        groups.setdefault(f"{ds}/{proto}/{model}", []).append(v["mae"])
    exp = ""
    PROTO = {"window": "Window", "record": "Record", "subject": "Subject", "reading": "Reading",
             "chrono0": "ChronoNoGap", "chrono": "Chrono"}
    for key, vals in groups.items():
        ds, proto, model = key.split("/")
        name = {"mendeley": "mend", "bigideas": "big"}[ds] + "Ladder" + PROTO[proto] + legacy.NAMES.get(model, "SubjMean")
        exp += legacy._macro(name, legacy._fmt(vals, 1))
    tex = to_latex_macros(store)
    got = {k: v for k, v in _macro_dict(tex).items() if "Ladder" in k}
    assert got == _macro_dict(exp)
    assert "mendLadderWindowOneD" in got and "bigLadderReadingSubjMean" in got


def test_latex_bootstrap_and_dataset_stats_match_legacy(tmp_path):
    rng = np.random.default_rng(3)
    boot = {"mae": 14.62, "mae_ci": [12.46, 17.46], "diff_vs_null": -0.0916, "diff_ci": [-0.641, 0.617],
            "p_better_than_null": 0.6208, "n_subjects": 23, "n_records": 67}
    store = ResultStore(tmp_path / "b.json", autosave=False)
    store.add(make_key("mendeley", "subject_loso", "fusion", 0), _metrics(rng))
    store.add("bootstrap/mendeley/subject_loso/fusion", boot)
    store.add("bootstrap/bigideas/chronological/1d", boot)
    # dataset stats from a prediction CSV, exactly as legacy dataset_stats() reads it
    df = pd.DataFrame({"record": [f"{s:02d}_{r:04d}" for s in range(1, 6) for r in range(1, 4)],
                       "subject": [f"{s:02d}" for s in range(1, 6) for _ in range(3)],
                       "y": rng.uniform(80, 200, 15).round(0), "pred": rng.uniform(80, 200, 15),
                       "n_windows": rng.integers(2, 5, 15)})
    store.write_predictions(make_key("mendeley", "subject_loso", "fusion", 0), df)
    csv = tmp_path / "legacy_pred.csv"
    df.to_csv(csv, index=False)
    exp = ""
    for key, pre in (("mendeley/fusion", "mendFusion"), ("bigideas/1d", "bigOneD")):
        v = boot
        exp += legacy._macro(f"{pre}Diff", f"{v['diff_vs_null']:+.1f}")
        exp += legacy._macro(f"{pre}DiffCI", f"[{v['diff_ci'][0]:+.1f}, {v['diff_ci'][1]:+.1f}]")
        exp += legacy._macro(f"{pre}MAECI", f"[{v['mae_ci'][0]:.1f}, {v['mae_ci'][1]:.1f}]")
        exp += legacy._macro(f"{pre}PBetter", f"{v['p_better_than_null']:.2f}")
    exp += legacy.dataset_stats(str(csv), "mend")
    got = _macro_dict(to_latex_macros(store))
    for k, v in _macro_dict(exp).items():
        assert got[k] == v, k


def test_latex_prefix_override_and_unknown_dataset_is_legal(tmp_path):
    store = ResultStore(tmp_path / "u.json", autosave=False)
    store.add(make_key("synthetic", "subject_kfold", "tiny_transformer", 0), _metrics(np.random.default_rng(4)))
    tex = to_latex_macros(store)
    names = list(_macro_dict(tex))
    assert all(macro_name(n) for n in names)
    assert "syntheticSubjectKfoldTinyTransformerMAE" in names
    tex2 = to_latex_macros(store, prefix_map={"synthetic/subject_kfold": "syn"})
    assert "synTinyTransformerMAE" in _macro_dict(tex2)
    assert tex.startswith("% Auto-generated")


def test_latex_degenerate_entries_do_not_crash(tmp_path):
    store = ResultStore(tmp_path / "d.json", autosave=False)
    store.add(make_key("synthetic", "personal", "mean", 0), {"n": 1, "mae": 3.0, "error": "degenerate"})
    store.add(make_key("synthetic", "personal", "ridge", 0), {"n": 0, "mae": None, "error": "degenerate"})
    got = _macro_dict(to_latex_macros(store))
    assert got["syntheticPersonalMeanMAE"] == "3.0" and got["syntheticPersonalMeanRMSE"] == "--"
    assert not any(k.startswith("syntheticPersonalRidge") for k in got)


def test_duplicate_macro_keeps_first(tmp_path, caplog):
    store = ResultStore(tmp_path / "dup.json", autosave=False)
    store.add("bootstrap/mendeley/subject_loso/fusion", {"mae": 1, "mae_ci": [0, 1], "diff_vs_null": 0.1,
                                                        "diff_ci": [0, 1], "p_better_than_null": 0.5})
    store.add("bootstrap/mendeley/fusion", {"mae": 2, "mae_ci": [1, 2], "diff_vs_null": 0.9,
                                           "diff_ci": [0, 1], "p_better_than_null": 0.1})
    with caplog.at_level("WARNING", logger="ppgbench.report"):
        got = _macro_dict(to_latex_macros(store))
    assert got["mendFusionDiff"] == "+0.1" and "emitted twice" in caplog.text


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def test_to_markdown(tmp_path):
    rng = np.random.default_rng(5)
    store = _store_from_legacy(tmp_path, _legacy_summary(rng, {"mean": 1, "fusion": 3}), "mendeley", "subject_loso")
    store.add("dataset/mendeley", {"n_records": 67, "n_subjects": 23, "n_windows": 1000, "glucose_min": 70,
                                   "glucose_max": 200, "glucose_mean": 120.0, "glucose_std": 20.0})
    store.add("bootstrap/mendeley/subject_loso/fusion", {"mae": 14.6, "mae_ci": [12.5, 17.5], "diff_vs_null": -0.1,
                                                        "diff_ci": [-0.6, 0.6], "p_better_than_null": 0.62,
                                                        "n_subjects": 23, "n_records": 67})
    md = to_markdown(store)
    assert "## mendeley / subject_loso" in md and "Fusion (ours) (`fusion`)" in md and " ± " in md
    assert "Cluster bootstrap" in md and "| synthetic" not in md and "## Datasets" in md and "| mendeley | 67 |" in md


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report.paper_style()
    return plt


def test_fig_clarke_zones(plt, tmp_path):
    y = np.array([60, 100, 150, 200, 250, 90.0]); pred = np.array([65, 180, 150, 60, 240, 250.0])
    fig, ax = plt.subplots()
    z = report.fig_clarke(ax, y, pred, title="t")
    assert set(z) == {f"zone_{k}_pct" for k in "ABCDE"} and sum(z.values()) == pytest.approx(100)
    assert z["zone_A_pct"] == pytest.approx(100 * 3 / 6) and z["zone_E_pct"] == pytest.approx(100 * 1 / 6)
    assert report.clarke_zone_pct(y, pred) == z
    paths = report.save_figure(fig, "clarke", tmp_path, formats=("png",))
    assert paths[0].exists() and paths[0].stat().st_size > 0


def test_fig_model_bars_and_ladder(plt, tmp_path):
    store = ResultStore(tmp_path / "f.json", autosave=False)
    rng = np.random.default_rng(6)
    for proto in ("window_random", "subject_loso"):
        for model in ("mean", "subject_mean", "1d", "fusion", "custom_model"):
            for seed in range(2):
                store.add(make_key("mendeley", proto, model, seed), _metrics(rng))
    fig, ax = plt.subplots()
    order = report.fig_model_bars(ax, aggregate(store, "mendeley", "subject_loso"), title="bars")
    assert order[0] == "mean" and order[-1] == "custom_model" and "fusion" in order
    assert [t.get_text() for t in ax.get_yticklabels()][0] == "Train mean"
    fig, ax = plt.subplots()
    report.fig_ladder(ax, store, "mendeley", ["window_random", "subject_loso"], ["windows", "subjects"])
    assert ax.get_legend() is not None
    plt.close("all")
    with pytest.raises(ValueError):
        report.fig_model_bars(plt.subplots()[1], {})


def test_fig_tracking_orders_by_time_then_record(plt):
    df = pd.DataFrame({"record": ["s_3", "s_1", "s_2"], "subject": ["s"] * 3, "y": [1.0, 2.0, 3.0],
                       "pred": [1.5, 2.5, 3.5], "time": [7200.0, 0.0, 3600.0]})
    fig, ax = plt.subplots()
    report.fig_tracking(ax, df, label="m")
    ref_line = ax.get_lines()[0]
    assert ref_line.get_xdata().tolist() == [0.0, 1.0, 2.0] and ref_line.get_ydata().tolist() == [2.0, 3.0, 1.0]
    fig, ax = plt.subplots()
    report.fig_tracking(ax, df.drop(columns="time"), spacing_min=30)
    assert ax.get_lines()[0].get_xdata().tolist() == [0.0, 0.5, 1.0]
    plt.close("all")


def test_make_figures_from_store(plt, tmp_path):
    store = ResultStore(tmp_path / "m.json", autosave=False)
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame({"record": [f"{i // 10:03d}_{i:05d}" for i in range(n)], "subject": [f"{i // 10:03d}" for i in range(n)],
                       "y": rng.uniform(80, 200, n), "pred": rng.uniform(80, 200, n), "n_windows": 2,
                       "time": np.arange(n) * 300.0})
    for model in ("subject_mean", "1d"):
        key = make_key("bigideas", "chronological", model, 0)
        store.add(key, _metrics(rng))
        store.write_predictions(key, df)
    written = report.make_figures(store, tmp_path / "figs", "bigideas", "chronological", model="fusion")
    names = {p.name for p in written}
    assert {"bigideas_chronological_models.pdf", "bigideas_chronological_clarke_1d.png",
            "bigideas_chronological_tracking_1d.pdf"} <= names                      # fell back to 1d
    assert all(p.exists() for p in written)
