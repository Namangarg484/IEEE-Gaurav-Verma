"""
End-to-end tests of ``python -m ppgbench`` on the deterministic ``synthetic``
dataset.  Everything runs in a few seconds: tiny corpus, ``--quick`` (3 folds)
and 1 epoch for the single network model.

The sibling modules (data / protocols / models / train / metrics / integrity)
are written by other owners; the tests skip cleanly while they are absent and
run for real once the package is complete.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _mod in ("ppgbench.data.registry", "ppgbench.protocols", "ppgbench.models.registry", "ppgbench.train",
             "ppgbench.metrics", "ppgbench.integrity", "ppgbench.core.features", "ppgbench.core.views"):
    pytest.importorskip(_mod)

from ppgbench import cli  # noqa: E402
from ppgbench.report import ResultStore, make_key  # noqa: E402

TINY = ["--dataset-arg", "n_subjects=4", "--dataset-arg", "records_per_subject=3",
        "--dataset-arg", "windows_per_record=3", "--dataset-arg", "seed=0"]


def _main(*argv: str) -> int:
    return cli.main(["-q", *argv])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_parse_kv():
    assert cli.parse_kv(["n=4", "root=/x/y", "frac=0.7", "subjects=['001','002']", "flag=True"]) == {
        "n": 4, "root": "/x/y", "frac": 0.7, "subjects": ["001", "002"], "flag": True}
    assert cli.parse_kv(None) == {}
    with pytest.raises(Exception):
        cli.parse_kv(["novalue"])


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "run" in capsys.readouterr().out
    assert _main("version") == 0
    assert capsys.readouterr().out.startswith("ppgbench ")
    assert cli.main(["version", "-q"]) == 0                 # verbosity flags accepted after the subcommand too


def test_datasets_and_models(capsys):
    assert _main("datasets") == 0
    assert "synthetic" in capsys.readouterr().out.split()
    assert _main("models") == 0
    out = capsys.readouterr().out
    rows = dict(line.split() for line in out.strip().splitlines())
    assert rows["mean"] == "classical" and rows["1d"] == "net" and "ridge" in rows
    assert _main("protocols") == 0
    out = capsys.readouterr().out
    assert "subject_loso" in out and "window_random" in out


# --------------------------------------------------------------------------- #
# run / ladder / audit / bootstrap / report
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quick_store(tmp_path_factory) -> Path:
    """One `run` with the spec's smoke-test configuration; shared by the later tests."""
    out = tmp_path_factory.mktemp("results") / "ppgbench.json"
    rc = _main("run", "--dataset", "synthetic", "--quick", "--models", "mean", "ridge", "1d",
               "--epochs", "1", "--device", "cpu", "--out", str(out), *TINY)
    assert rc == 0
    return out


def test_run_quick_end_to_end(quick_store, capsys):
    store = ResultStore(quick_store)
    keys = [k for k in store.keys() if k.startswith("synthetic/subject_loso/")]
    assert keys == [make_key("synthetic", "subject_loso", m, 0) for m in ("mean", "ridge", "1d")]
    for k in keys:
        m = store[k]
        assert m["quick"] is True and m["folds"] == cli.QUICK_FOLDS and m["protocol"] == "subject_loso"
        assert m["n"] > 0 and m["mae"] is not None and m["mae"] >= 0 and "zone_A_pct" in m
        assert m["level"] == "subject" and "per_subject" in m and len(m["subjects"]) == 4
        df = store.read_predictions(k)
        assert df is not None and {"record", "subject", "y", "pred", "n_windows", "time"} <= set(df.columns)
        assert len(df) == m["n"] and df["record"].is_unique            # one row per reference measurement
        assert set(df["subject"]) <= set(m["subjects"]) and len(set(df["subject"])) == cli.QUICK_FOLDS
    net = store[make_key("synthetic", "subject_loso", "1d", 0)]
    assert net["epochs"] == 1 and net["kind"] == "net" and net["device"] == "cpu"
    assert "epochs" not in store[make_key("synthetic", "subject_loso", "mean", 0)]
    ds = store["dataset/synthetic"]
    assert ds["n_subjects"] == 4 and ds["n_records"] == 12 and ds["n_windows"] == 36 and ds["has_time"] is True
    raw = json.loads(quick_store.read_text())
    assert set(raw) == set(store.keys())


def test_run_prints_one_line_per_model_seed(tmp_path, capsys):
    out = tmp_path / "s.json"
    assert _main("run", "--dataset", "synthetic", "--quick", "--models", "mean", "ridge", "--out", str(out), *TINY) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("RESULT ")]
    assert len(lines) == 2 and "synthetic/subject_loso/mean/seed0" in lines[0] and "MAE" in lines[0]


def test_run_multiple_seeds_only_for_nets(tmp_path):
    out = tmp_path / "s.json"
    assert _main("run", "--dataset", "synthetic", "--quick", "--protocol", "subject_kfold", "--protocol-arg", "n_splits=2",
                 "--models", "mean", "1d", "--seeds", "0", "1", "--epochs", "1", "--device", "cpu",
                 "--out", str(out), *TINY) == 0
    store = ResultStore(out)
    assert len(store.runs(model="mean")) == 1 and len(store.runs(model="1d")) == 2
    assert store.runs(model="1d")[0][1]["folds"] == 2


def test_run_rejects_unknown_model(tmp_path):
    with pytest.raises(SystemExit):
        _main("run", "--dataset", "synthetic", "--quick", "--models", "no_such_model", "--out", str(tmp_path / "x.json"), *TINY)


def test_ladder(tmp_path):
    out = tmp_path / "ladder.json"
    assert _main("ladder", "--dataset", "synthetic", "--quick", "--models", "mean", "subject_mean",
                 "--protocols", "window_random", "record_random", "subject_loso", "--out", str(out), *TINY) == 0
    store = ResultStore(out)
    assert store.experiments() == [("synthetic", "window_random"), ("synthetic", "record_random"),
                                   ("synthetic", "subject_loso")]
    # the leaky subject-mean predictor must beat the null when records of a subject leak, and tie under LOSO
    g = {p: store.groups("synthetic", p) for p, _ in [(e[1], None) for e in store.experiments()]}
    assert g["record_random"]["subject_mean"][0]["mae"] < g["record_random"]["mean"][0]["mae"]
    assert g["subject_loso"]["subject_mean"][0]["mae"] == pytest.approx(g["subject_loso"]["mean"][0]["mae"])


def test_audit_passes_and_detects_leak(tmp_path, capsys):
    rc = _main("audit", "--dataset", "synthetic", "--protocol", "subject_loso", "--json", str(tmp_path / "a.json"), *TINY)
    assert rc == 0
    rep = json.loads((tmp_path / "a.json").read_text())
    assert rep["no_leak"] is True and rep["level"] == "subject" and "audit" in rep and "leak_report" in rep
    # record-wise folds share subjects: asserting the subject level must fail with exit code 1
    rc = _main("audit", "--dataset", "synthetic", "--protocol", "record_random", "--level", "subject", *TINY)
    assert rc == 1
    assert "LEAK DETECTED" in capsys.readouterr().out


def test_bootstrap(quick_store, capsys):
    assert _main("bootstrap", "--store", str(quick_store), "--dataset", "synthetic", "--protocol", "subject_loso",
                 "-B", "50", "--seed", "0") == 0
    store = ResultStore(quick_store)
    v = store["bootstrap/synthetic/subject_loso/1d"]
    assert v["null_model"] == "mean" and v["B"] == 50 and len(v["mae_ci"]) == 2 and 0 <= v["p_better_than_null"] <= 1
    assert "bootstrap/synthetic/subject_loso/mean" not in store
    out = capsys.readouterr().out
    assert "ridge" in out and "1d" in out
    with pytest.raises(SystemExit):
        _main("bootstrap", "--store", str(quick_store), "--dataset", "synthetic", "--protocol", "subject_loso",
              "--null", "fusion")


def test_report_exports(quick_store, tmp_path, capsys):
    tex, md, figs = tmp_path / "m.tex", tmp_path / "s.md", tmp_path / "figs"
    assert _main("report", "--store", str(quick_store), "--latex", str(tex), "--markdown", str(md),
                 "--figures", str(figs), "--model", "1d", "--prefix", "synthetic/subject_loso=syn") == 0
    t = tex.read_text()
    assert "\\newcommand{\\synOneDMAE}" in t and "\\newcommand{\\synMeanN}" in t
    assert "\\newcommand{\\syntheticNRecords}{12}" in t and "\\newcommand{\\syntheticLadderSubjectRidge}" in t
    assert "\\newcommand{\\synOneDDiffCI}" in t                        # bootstrap entry from the previous test
    m = md.read_text()
    assert "## synthetic / subject_loso" in m and "`1d`" in m
    assert (figs / "synthetic_subject_loso_models.pdf").exists() and (figs / "synthetic_subject_loso_clarke_1d.png").exists()
    # no outputs requested -> markdown on stdout
    assert _main("report", "--store", str(quick_store)) == 0
    assert "## synthetic / subject_loso" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _main("report", "--store", str(tmp_path / "missing.json"))


def test_python_m_entry_point():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (os.environ.get("PYTHONPATH", ""), str(ROOT)) if p))
    r = subprocess.run([sys.executable, "-m", "ppgbench", "version"], capture_output=True, text=True, env=env,
                       cwd=str(ROOT), timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ppgbench ")
