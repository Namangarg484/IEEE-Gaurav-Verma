"""
bootstrap_ci.py
===============
Cluster-bootstrap (resampling SUBJECTS) 95 % confidence intervals for each
model's MAE and for its paired MAE difference against the null model, from
the out-of-fold prediction files in results/.

    python bootstrap_ci.py            # -> results/bootstrap_ci.json (+ printed table)

The paired difference is computed on the same resampled subjects for both
models, so it respects the pairing and the within-subject correlation.  A CI
of the difference that excludes zero is reported as "significant".
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

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


def cluster_bootstrap(df_model, df_null, rng):
    """Returns (mae, lo, hi, diff, dlo, dhi, p_better)."""
    m = df_model.merge(df_null, on=["record", "subject"], suffixes=("", "_null"))
    subs = m.subject.unique()
    by = {s: g for s, g in m.groupby("subject")}
    e_m = {s: np.abs(g.pred - g.y).to_numpy() for s, g in by.items()}
    e_n = {s: np.abs(g.pred_null - g.y).to_numpy() for s, g in by.items()}
    maes, diffs = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.choice(subs, len(subs), replace=True)
        em = np.concatenate([e_m[s] for s in pick]); en = np.concatenate([e_n[s] for s in pick])
        maes[b] = em.mean(); diffs[b] = em.mean() - en.mean()
    mae = float(np.abs(m.pred - m.y).mean()); diff = mae - float(np.abs(m.pred_null - m.y).mean())
    return dict(mae=mae, mae_ci=[float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
                diff_vs_null=diff, diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                p_better_than_null=float((diffs < 0).mean()), n_subjects=int(len(subs)), n_records=int(len(m)))


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
            pat = pat_fmt.format(model)
            dfm = load(pat)
            if dfm is None:
                continue
            out[f"{exp}/{model}"] = cluster_bootstrap(dfm, null, rng)      # seed-ensemble (mean prediction per record)
            files = sorted(glob.glob(os.path.join(RES, pat)))
            if len(files) > 1:                                              # per-seed entries as well
                for f in files:
                    k = f.rsplit("_seed", 1)[1].split(".")[0]
                    out[f"{exp}/{model}/seed{k}"] = cluster_bootstrap(load(os.path.basename(f)), null, np.random.default_rng(int(k)))
    json.dump(out, open(os.path.join(RES, "bootstrap_ci.json"), "w"), indent=2)
    print(f"{'experiment/model':22s} {'MAE [95% CI]':>22s} {'diff vs null [95% CI]':>26s} {'P(better)':>10s}")
    for k, v in out.items():
        print(f"{k:22s} {v['mae']:6.2f} [{v['mae_ci'][0]:5.2f}, {v['mae_ci'][1]:5.2f}]   "
              f"{v['diff_vs_null']:+6.2f} [{v['diff_ci'][0]:+5.2f}, {v['diff_ci'][1]:+5.2f}]   {v['p_better_than_null']:8.3f}")


if __name__ == "__main__":
    main()
