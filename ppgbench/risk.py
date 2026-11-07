"""A-priori identity-leak risk: predict how much a leaky split can inflate a metric, before training.

Under a split that lets a subject's other samples into training, a model can
lower its error by recognising the subject and recalling the subject's label
level.  Two quantities, both computable from the labels, the subject ids, the
inputs and the folds alone, bound how much:

* **potential** ``P = MAE(train-mean null) - MAE(subject oracle)``, where the
  subject oracle predicts each test sample by the mean label of the same
  subject's training samples (falling back to the training mean);
* **identifiability** ``a(X)``: accuracy of 1-nearest-neighbour subject
  identification from the standardised inputs ``X`` under the same folds.

The predicted inflation is ``I_hat = a * P``: what a model that identified
subjects as well as the nearest-neighbour test, and then recalled their level,
would gain.  Flexible models approach it; models that cannot express
per-subject offsets (e.g. linear models) realise less.  ``X`` must be the
exact inputs the model will receive -- auxiliary inputs such as demographics
can identify subjects without carrying any information about the target.

Example
-------
>>> from ppgbench.risk import leak_risk
>>> r = leak_risk(y, subject, X, folds)          # folds: list of (train_idx, test_idx)
>>> r["predicted_inflation"], r["potential"], r["identifiability"]
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = ["icc1", "potential", "identifiability", "leak_risk"]


def icc1(y: Sequence[float], groups: Sequence[Any]) -> float:
    """One-way random-effects ICC(1) of ``y`` grouped by ``groups`` (unbalanced groups allowed)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)})
    k = df.groupby("g").size()
    n, a = len(df), len(k)
    if a < 2 or n <= a:
        return float("nan")
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    msb = (df.groupby("g").y.mean().sub(df.y.mean()) ** 2 * k).sum() / (a - 1)
    msw = df.groupby("g").y.var(ddof=1).fillna(0).mul(k - 1).sum() / (n - a)
    den = msb + (k0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else 0.0


def potential(y: Sequence[float], subject: Sequence[Any], folds) -> dict[str, float]:
    """MAE of the train-mean null, of the subject oracle, and their difference ``P``."""
    y, s = np.asarray(y, float), np.asarray(subject)
    p_null, p_or = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in folds:
        mu = pd.Series(y[tr]).groupby(s[tr]).mean()
        p_null[te] = y[tr].mean()
        p_or[te] = pd.Series(s[te]).map(mu).fillna(y[tr].mean()).to_numpy()
    ok = ~np.isnan(p_null)
    m_null, m_or = float(np.abs(p_null[ok] - y[ok]).mean()), float(np.abs(p_or[ok] - y[ok]).mean())
    return {"mae_null": m_null, "mae_oracle": m_or, "potential": m_null - m_or}


def identifiability(X: np.ndarray, subject: Sequence[Any], folds) -> float:
    """Fraction of test samples whose subject a 1-NN classifier on standardised ``X`` recovers."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    X, s = np.asarray(X, float), np.asarray(subject).astype(str)
    if X.ndim == 1:
        X = X[:, None]
    hit = tot = 0
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = KNeighborsClassifier(1, algorithm="brute").fit(sc.transform(X[tr]), s[tr])
        hit += int((clf.predict(sc.transform(X[te])) == s[te]).sum()); tot += len(te)
    return hit / tot if tot else float("nan")


def leak_risk(y: Sequence[float], subject: Sequence[Any], X: np.ndarray, folds) -> dict[str, float]:
    """Predicted identity-leak inflation ``a * P`` with its components (and the label ICC)."""
    pot = potential(y, subject, folds)
    a = identifiability(X, subject, folds)
    return {**pot, "identifiability": a, "predicted_inflation": a * pot["potential"], "icc": icc1(y, subject)}
