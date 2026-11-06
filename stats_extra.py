"""
stats_extra.py
==============
Inferential statistics beyond bootstrap_ci.py, all from results/*.csv:

  * subject-cluster bootstrap 95 % CI for Pearson r and for the median
    within-subject r, per model;
  * Wilcoxon signed-rank test on per-measurement |error| model vs null;
  * minimum detectable effect (half-width of the paired-difference CI of the
    primary comparison);
  * leverage analysis on Mendeley: r with and without recordings > 140 mg/dL;
  * two-sided sign test on 'worse than the subject mean for k of n subjects'.

    python stats_extra.py      # -> results/stats_extra.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
B = 5000


def load(pattern):
    files = sorted(glob.glob(os.path.join(RES, pattern)))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    df["subject"] = df.subject.astype(str).str.zfill(3)
    return df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))


def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan


def r_ci(df, rng):
    """Subject-cluster bootstrap CIs of pooled r and of the median within-subject r (numpy only)."""
    subs = df.subject.unique()
    Y = {s: g.y.to_numpy() for s, g in df.groupby("subject")}
    P = {s: g.pred.to_numpy() for s, g in df.groupby("subject")}
    wr = np.array([_r(Y[s], P[s]) for s in subs])                 # per-subject r (NaN if undefined)
    idx = {s: i for i, s in enumerate(subs)}
    rs, ws = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.choice(subs, len(subs), replace=True)
        y = np.concatenate([Y[s] for s in pick]); p = np.concatenate([P[s] for s in pick])
        rs[b] = _r(y, p)
        w = wr[[idx[s] for s in pick]]
        ws[b] = np.nanmedian(w) if np.isfinite(w).any() else np.nan
    return (float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5)),
            float(np.nanpercentile(ws, 2.5)) if np.isfinite(ws).any() else None,
            float(np.nanpercentile(ws, 97.5)) if np.isfinite(ws).any() else None)


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for exp, null_pat, pat_fmt, models in (
        ("mendeley", "subject_cv_mean_seed0.csv", "subject_cv_{}_seed*.csv",
         ["ridge", "demo", "ridge+demo", "gbm", "gbm+demo", "1d", "2d", "3d", "fusion", "resnet1d", "cnnlstm", "papagei", "anyppg", "pulseppg"]),
        ("bigideas", "bigideas_within_mean_seed0.csv", "bigideas_within_{}_seed*.csv",
         ["ridge", "1d", "2d", "3d", "fusion", "tod", "persist", "ridge_res", "1d_res", "2d_res", "3d_res", "fusion_res", "papagei", "anyppg", "pulseppg", "papagei_res", "anyppg_res", "pulseppg_res"]),
        ("stuba", "subject_cv_stuba_IR1_mean_seed0.csv", "subject_cv_stuba_IR1_{}_seed*.csv",
         ["ridge", "gbm", "1d", "2d", "3d", "fusion", "papagei", "anyppg", "pulseppg"]),
    ):
        null = load(null_pat)
        if null is None:
            continue
        for model in models:
            dfm = load(pat_fmt.format(model))
            if dfm is None:
                continue
            m = dfm.merge(null, on=["record", "subject"], suffixes=("", "_null"))
            e_m, e_n = np.abs(m.pred - m.y), np.abs(m.pred_null - m.y)
            d = {"n": int(len(m)), "pearson_r": _r(m.y.to_numpy(), m.pred.to_numpy())}
            ci = r_ci(m, rng)
            d["r_ci"], d["within_r_ci"] = list(ci[:2]), list(ci[2:])
            w = [_r(g.y.to_numpy(), g.pred.to_numpy()) for _, g in m.groupby("subject")]
            d["within_r_median"] = float(np.nanmedian(w)) if np.isfinite(w).any() else None
            try:
                d["wilcoxon_p"] = float(wilcoxon(e_m, e_n).pvalue)
            except ValueError:
                d["wilcoxon_p"] = None
            per = m.groupby("subject").apply(lambda g: float((np.abs(g.pred - g.y)).mean() - (np.abs(g.pred_null - g.y)).mean()),
                                             include_groups=False)
            worse = int((per > 0).sum()); n = int(len(per))
            d["subjects_worse"], d["subjects_n"] = worse, n
            d["sign_test_p"] = float(binomtest(worse, n, 0.5).pvalue)
            if exp == "mendeley":
                lo = m[m.y <= 140]
                d["r_without_high"] = _r(lo.y.to_numpy(), lo.pred.to_numpy()); d["n_high"] = int((m.y > 140).sum())
            out[f"{exp}/{model}"] = d
    # minimum detectable effect from bootstrap_ci.json (primary comparison: 1d vs null)
    p = os.path.join(RES, "bootstrap_ci.json")
    if os.path.exists(p):
        ci = json.load(open(p))
        for exp in ("mendeley", "bigideas", "stuba"):
            for model, suffix in (("1d", ""), ("1d_res", "_res")):
                k = f"{exp}/{model}"
                if k in ci:
                    lo, hi = ci[k]["diff_ci"]
                    out[f"{exp}/mde{suffix}"] = {"half_width": float((hi - lo) / 2), "n_records": ci[k]["n_records"]}
    json.dump(out, open(os.path.join(RES, "stats_extra.json"), "w"), indent=2)
    for k, v in out.items():
        if "pearson_r" in v:
            print(f"{k:22s} r {v['pearson_r']:+.3f} CI [{v['r_ci'][0]:+.2f},{v['r_ci'][1]:+.2f}]  within-r {v['within_r_median']}  "
                  f"wilcoxon p {v['wilcoxon_p']}  worse {v['subjects_worse']}/{v['subjects_n']} (p={v['sign_test_p']:.3f})"
                  + (f"  r(y<=140) {v['r_without_high']:+.3f}" if "r_without_high" in v else ""))
        else:
            print(k, v)


if __name__ == "__main__":
    main()
