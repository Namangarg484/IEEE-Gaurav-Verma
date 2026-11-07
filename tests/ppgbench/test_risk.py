"""Tests for the a-priori identity-leak risk predictor."""
import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from ppgbench.risk import icc1, identifiability, leak_risk, potential


def _cohort(n_subj=20, n_rec=10, between=10.0, within=1.0, fingerprint=True, seed=0):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), n_rec)
    level = rng.normal(100, between, n_subj)
    y = level[subj] + rng.normal(0, within, len(subj))
    code = rng.normal(0, 1, (n_subj, 4))
    X = (code[subj] if fingerprint else rng.normal(0, 1, (n_subj, 4))[rng.integers(0, n_subj, len(subj))]) \
        + rng.normal(0, 0.05, (len(subj), 4))
    return y, subj, X


def test_icc_extremes():
    y, s, _ = _cohort(between=10, within=0.01)
    assert icc1(y, s) > 0.99
    y2, s2, _ = _cohort(between=0.0, within=5)
    assert abs(icc1(y2, s2)) < 0.2


def test_identifiable_and_clustered_gives_large_risk():
    y, s, X = _cohort()
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    r = leak_risk(y, s, X, folds)
    assert r["identifiability"] > 0.95
    assert r["potential"] > 5 and abs(r["predicted_inflation"] - r["identifiability"] * r["potential"]) < 1e-12


def test_non_identifying_inputs_give_small_risk():
    y, s, X = _cohort(fingerprint=False)
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    r = leak_risk(y, s, X, folds)
    assert r["identifiability"] < 0.25
    assert r["predicted_inflation"] < 0.3 * r["potential"]


def test_subject_wise_folds_have_zero_potential():
    y, s, X = _cohort()
    folds = list(GroupKFold(5).split(X, y, s))
    assert abs(potential(y, s, folds)["potential"]) < 1e-9          # oracle falls back to the train mean
    assert identifiability(X, s, folds) == 0.0                         # test subjects never seen in training
