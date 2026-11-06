"""
Regression tests from the adversarial leakage review of ``ppgbench``.

Each test pins one way in which information from the test folds -- or a
weaker split than the one a result claims -- could have reached a *stored or
reported* number without tripping the no-leak machinery:

* ``Experiment.run`` honoured ``--protocol-arg check=False`` and stored the
  unverified folds under the protocol's declared level;
* a run with weakened protocol parameters (``gap_s=0``, ``train_frac=0.5``)
  was stored under the canonical key, silently replacing the honest result
  and inheriting its LaTeX macro name;
* ``python -m ppgbench audit`` asserted the ``time`` level with ``gap_s=0``
  while the integrity report used the protocol *default* gap, so a split
  built with ``--protocol-arg gap_s=...`` was reported inconsistently and a
  builder that ignored its gap could not be caught;
* ``leak_report`` treated a NaN reference time as "infinitely far from
  training" and reported ``passes["time"] = True``;
* ``ProbeModel`` keyed its embedding cache on ``id(ws.X)`` alone, which a
  recycled array id could alias.

Everything runs on the ``synthetic`` dataset with classical models only, so
the whole module takes a few seconds and never trains a network.
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ppgbench import cli  # noqa: E402
from ppgbench import integrity as I  # noqa: E402
from ppgbench import protocols as P  # noqa: E402
from ppgbench.core.windowset import WindowSet  # noqa: E402
from ppgbench.data.registry import load_dataset  # noqa: E402
from ppgbench.protocols import (  # noqa: E402
    LeakError, chronological_split, declared_level, fold_fingerprint, leak_report, make_folds, protocol_variant,
    split_protocol_variant,
)
from ppgbench.report import ResultStore, base_protocol, make_key, protocol_token, to_latex_macros  # noqa: E402

#: 4 subjects x 36 CGM-like readings (3 h at 300 s): the 1-h guard gap of ``chronological`` leaves training data
DS = dict(n_subjects=4, records_per_subject=36, windows_per_record=2)
TINY = [a for k, v in DS.items() for a in ("--dataset-arg", f"{k}={v}")]


@pytest.fixture(scope="module")
def ws() -> WindowSet:
    return load_dataset("synthetic", **DS)


def _experiment(tmp_path):
    store = ResultStore(tmp_path / "store.json")
    return cli.Experiment("synthetic", DS, store, "cpu"), store


def _classical(exp: cli.Experiment, protocol: str, model: str, fold_kwargs=None):
    return exp.run(protocol, model, "classical", 0, 1, 16, 1e-3, False, fold_kwargs, {})


@pytest.fixture
def broken_protocol(monkeypatch) -> str:
    """A registered protocol whose single fold leaks at every level (train = every window)."""
    def broken(w, seed=0):
        n = len(w)
        return [(np.arange(n), np.arange(n // 2))]
    monkeypatch.setitem(P.PROTOCOLS, "broken", P.ProtocolSpec("broken", broken, "subject", "leaky on purpose", {}, False))
    return "broken"


# --------------------------------------------------------------------------- #
# 1. the self-check cannot be switched off for a stored result
# --------------------------------------------------------------------------- #
def test_cli_run_ignores_check_false_and_refuses_leaky_folds(tmp_path, broken_protocol, caplog):
    exp, store = _experiment(tmp_path)
    with caplog.at_level(logging.WARNING, logger="ppgbench.cli"):
        with pytest.raises(LeakError):
            _classical(exp, broken_protocol, "mean", {"check": False})
    assert any("check" in r.getMessage() for r in caplog.records)
    with pytest.raises(LeakError):
        _classical(exp, broken_protocol, "mean")
    assert not [k for k in store.keys() if "/broken/" in k]            # nothing was stored for the leaky split


def test_run_stores_fold_provenance(tmp_path):
    exp, store = _experiment(tmp_path)
    key, m = _classical(exp, "subject_loso", "mean")
    assert key == make_key("synthetic", "subject_loso", "mean", 0)
    assert m["leak_checked"] is True and m["level"] == "subject" and m["gap_s"] is None
    assert m["protocol"] == "subject_loso" and m["base_protocol"] == "subject_loso" and m["variant"] == ""
    assert m["fold_fingerprint"] == fold_fingerprint(make_folds(exp.ws, "subject_loso"))
    assert m["n_folds_total"] == 4 and m["folds"] == 4 and m["protocol_params"] == {}
    key, m = _classical(exp, "chronological", "subject_mean")
    assert m["level"] == "time" and m["gap_s"] == 3600.0
    assert m["protocol_params"] == {"train_frac": 0.7, "gap_s": 3600.0} and m["fold_kwargs"] == {}
    assert store[key]["fold_fingerprint"] == m["fold_fingerprint"]


# --------------------------------------------------------------------------- #
# 2. weakened protocol parameters never overwrite the canonical result
# --------------------------------------------------------------------------- #
def test_protocol_variant_spelling():
    assert protocol_variant("chronological", {}) == ""
    assert protocol_variant("chronological", {"gap_s": 3600.0, "train_frac": 0.7}) == ""     # defaults: no variant
    assert protocol_variant("chronological", {"gap_s": 0}) == "_gap0"                         # legacy _gap0 tag
    assert protocol_variant("chronological", {"gap_s": 1800, "train_frac": 0.5}) == "_cut50_gap1800"   # signature order
    assert protocol_variant("chrono", {"gap_s": 0}) == "_gap0"                                # legacy alias accepted
    assert protocol_variant("subject_kfold", {"n_splits": 2}) == "_k2"
    assert protocol_variant("personal", {"min_train": 32, "gap_s": 3600.0}) == "_mintrain32"
    assert protocol_variant("window_random", {"check": False}) == ""                          # not a protocol parameter
    assert split_protocol_variant("chronological_cut50_gap0") == ("chronological", "_cut50_gap0")
    assert split_protocol_variant("chronological_nogap") == ("chronological_nogap", "")
    assert split_protocol_variant("subject_kfold_k2") == ("subject_kfold", "_k2")
    assert split_protocol_variant("no_such_protocol_k2") == ("no_such_protocol_k2", "")
    assert declared_level("chronological_gap0") == "time" and declared_level("subject_kfold_k2") == "subject"
    assert cli.protocol_level("chronological_cut50_gap0") == "time"
    assert base_protocol("chronological_cut50_gap0") == "chronological"
    assert base_protocol("chronological_nogap") == "chronological_nogap" and base_protocol("subject_kfold") == "subject_kfold"
    assert protocol_token("chronological_gap0") == "ChronoGapZero"
    assert protocol_token("record_random_k3", "bigideas") == "ReadingKThree"


def test_weakened_protocol_parameters_get_their_own_key_and_macros(tmp_path):
    exp, store = _experiment(tmp_path)
    k_honest, m_honest = _classical(exp, "chronological", "subject_mean")
    k_nogap, m_nogap = _classical(exp, "chronological", "subject_mean", {"gap_s": 0})
    k_cut, m_cut = _classical(exp, "chronological", "subject_mean", {"gap_s": 0, "train_frac": 0.5})
    assert k_honest == "synthetic/chronological/subject_mean/seed0"
    assert k_nogap == "synthetic/chronological_gap0/subject_mean/seed0"
    assert k_cut == "synthetic/chronological_cut50_gap0/subject_mean/seed0"
    # the canonical entry is untouched by the later, weaker runs
    assert store[k_honest]["mae"] == m_honest["mae"] and store[k_honest]["fold_kwargs"] == {}
    assert store[k_honest]["gap_s"] == 3600.0 and store[k_nogap]["gap_s"] == 0.0
    assert m_nogap["protocol"] == "chronological_gap0" and m_nogap["base_protocol"] == "chronological"
    assert m_nogap["variant"] == "_gap0" and m_nogap["level"] == "time"
    assert m_nogap["fold_fingerprint"] != m_honest["fold_fingerprint"] != m_cut["fold_fingerprint"]
    assert m_cut["protocol_params"] == {"train_frac": 0.5, "gap_s": 0}
    assert [p for _, p in store.experiments()] == ["chronological", "chronological_gap0", "chronological_cut50_gap0"]
    # ... and so are the LaTeX macros: three distinct names, the canonical one carrying the honest number
    tex = to_latex_macros(store)
    assert f"\\newcommand{{\\syntheticChronoSubjMeanMAE}}{{{m_honest['mae']:.1f}}}" in tex
    assert "\\newcommand{\\syntheticChronoGapZeroSubjMeanMAE}" in tex
    assert "\\newcommand{\\syntheticChronoCutFiveZeroGapZeroSubjMeanMAE}" in tex
    assert tex.count("SubjMeanMAE}") == 3


def test_bootstrap_null_model_follows_the_base_protocol(tmp_path):
    exp, store = _experiment(tmp_path)
    for model in ("subject_mean", "mean"):
        _classical(exp, "chronological", model, {"gap_s": 0})
    rc = cli.main(["-q", "bootstrap", "--store", str(store.path), "--dataset", "synthetic",
                   "--protocol", "chronological_gap0", "-B", "20"])
    assert rc == 0
    store.load()
    v = store["bootstrap/synthetic/chronological_gap0/mean"]
    assert v["null_model"] == "subject_mean" and v["n_subjects"] == DS["n_subjects"]


# --------------------------------------------------------------------------- #
# 3. `audit` verifies with the gap actually in force
# --------------------------------------------------------------------------- #
def test_audit_uses_the_gap_in_force(tmp_path, capsys):
    out = tmp_path / "audit.json"
    rc = cli.main(["-q", "audit", "--dataset", "synthetic", "--protocol", "chronological",
                   "--protocol-arg", "gap_s=600", "--json", str(out), *TINY])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["no_leak"] is True and rep["gap_s"] == 600.0 and rep["fold_kwargs"] == {"gap_s": 600}
    assert rep["audit"]["ok"] is True and rep["audit"]["leak"]["error"] is None and rep["audit"]["leak"]["gap_s"] == 600.0
    assert rep["leak_report"]["gap_s"] == 600.0 and rep["leak_report"]["declared_level_ok"] is True
    assert rep["leak_report"]["totals"]["min_test_train_distance_s"] > 600.0
    assert rep["fingerprint"] == rep["leak_report"]["fingerprint"] == rep["audit"]["leak"]["fingerprint"]
    assert "gap_s=600" in capsys.readouterr().out


def test_audit_catches_a_builder_that_ignores_its_gap(tmp_path, monkeypatch, capsys):
    def sloppy(w, seed=0, *, train_frac=0.7, gap_s=3600.0):
        return [chronological_split(w, train_frac, 0.0)]            # the guard gap is silently dropped
    monkeypatch.setitem(P.PROTOCOLS, "sloppy", P.ProtocolSpec(
        "sloppy", sloppy, "time", "ignores gap_s", {"train_frac": 0.7, "gap_s": 3600.0}, False))
    out = tmp_path / "audit.json"
    rc = cli.main(["-q", "audit", "--dataset", "synthetic", "--protocol", "sloppy", "--json", str(out), *TINY])
    rep = json.loads(out.read_text())
    assert rc == 1 and rep["no_leak"] is False and "gap_s=3600" in rep["leak_error"]
    assert rep["audit"]["ok"] is False and rep["audit"]["leak"]["error"]
    assert rep["leak_report"]["declared_level_ok"] is False
    assert "LEAK DETECTED" in capsys.readouterr().out
    # the same builder with an explicit gap_s=0 is what it claims to be
    rc = cli.main(["-q", "audit", "--dataset", "synthetic", "--protocol", "sloppy", "--protocol-arg", "gap_s=0",
                   "--json", str(out), *TINY])
    assert rc == 0 and json.loads(out.read_text())["gap_s"] == 0.0


# --------------------------------------------------------------------------- #
# 4. a NaN reference time can never pass the time level
# --------------------------------------------------------------------------- #
def test_leak_report_does_not_treat_nan_times_as_far_away(ws):
    folds = make_folds(ws, "chronological")
    t = ws.time.copy()
    t[folds[0][1][:2]] = np.nan
    w = WindowSet(ws.X, ws.y, ws.subject, ws.record, t, ws.meta, ws.fs, "nan-times")
    rep = leak_report(w, folds, "chronological")
    assert rep["has_time"] is True and rep["time_finite"] is False
    assert rep["totals"]["time_nonfinite_windows"] == 2
    assert rep["passes"]["time"] is None and rep["declared_level_ok"] is None
    with pytest.raises(ValueError, match="NaN"):
        P.assert_no_leak(w, folds, "time", gap_s=3600.0)
    audit = I.audit(w, folds, "chronological")                       # reported, not raised
    assert audit["ok"] is False and audit["leak"]["error"] and "NaN" in audit["leak"]["error"]
    assert audit["leak"]["time_finite"] is False and audit["leak"]["gap_s"] == 3600.0
    clean = leak_report(ws, folds, "chronological")
    assert clean["time_finite"] is True and clean["totals"]["time_nonfinite_windows"] == 0
    assert clean["passes"]["time"] is True and clean["declared_level_ok"] is True


# --------------------------------------------------------------------------- #
# 5. the probe embedding cache cannot alias a recycled array id
# --------------------------------------------------------------------------- #
def test_probe_embedding_cache_pins_the_array(ws):
    from ppgbench.models.pretrained import ProbeModel
    m = ProbeModel("identity")
    Z = m.embeddings(ws)
    ((cached_X, cached_Z),) = m._cache.values()
    assert cached_X is ws.X and cached_Z is Z
    assert m.embeddings(ws) is Z                                      # same array object: served from the cache
    other = WindowSet(ws.X.copy(), ws.y, ws.subject, ws.record, ws.time, ws.meta, ws.fs, "copy")
    assert m.embeddings(other) is not Z                               # another array of the same shape: recomputed
