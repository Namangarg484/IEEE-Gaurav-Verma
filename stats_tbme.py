"""
stats_tbme.py
=============
Two analyses requested in review of the TBME manuscript, computed from saved
out-of-fold predictions (no retraining):

1. Seed-ensemble decisions.  The decision rule is applied, for every
   multi-seed model, to the seed-ensemble prediction (mean of the seeds'
   predictions per record), exactly as for the glucose primary comparison.
   Covers the PPG-BP 1-D CNN and re-audit protocols B and C (protocol A has no
   common test set across seeds, because each seed draws its own noise).

2. Power at glucose-cohort sizes.  Would the rule have detected the
   blood-pressure signal if the evaluation cohort were as small as the
   glucose cohorts?  Patients are drawn without replacement from the honest
   (subject-wise) out-of-fold predictions and the rule is re-applied:
     * VitalDB, 23 patients x 3 segments (Kermani-Esmaeili-like design),
     * VitalDB, 16 patients x all their segments (BIG IDEAs-like),
     * PPG-BP, 23 subjects x 1 reading.
   The models were trained on the full cohort, so this measures the power of
   the *evaluation*, not of training on a small cohort.

    python stats_tbme.py          # -> results/stats_tbme.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from validate_detectors import cluster_ci

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "stats_tbme.json")


def ensemble_decision(files, null_file, key="pred", null_key="pred", on=("record", "subject"), rng=None):
    dfs = [pd.read_csv(f) for f in files]
    m = dfs[0][list(on) + ["y"]].copy()
    m["pred"] = np.mean([d.set_index(list(on)).loc[list(map(tuple, m[list(on)].to_numpy()))][key].to_numpy() for d in dfs], 0) \
        if len(on) > 1 else np.mean([d[key].to_numpy() for d in dfs], 0)
    n = pd.read_csv(null_file)
    m = m.merge(n[list(on) + [null_key]].rename(columns={null_key: "pred_null"}), on=list(on))
    d, ci = cluster_ci(m.y.to_numpy(), m.pred.to_numpy(), m.pred_null.to_numpy(), m.subject.astype(str).to_numpy(),
                       rng or np.random.default_rng(0), B=5000)
    return {"n_seeds": len(files), "mae": float(np.abs(m.pred - m.y).mean()), "diff_vs_null": d,
            "diff_ci": [float(ci[0]), float(ci[1])], "n_records": int(len(m))}


def reaudit_ensemble(proto, rng):
    files = sorted(glob.glob(os.path.join(RES, f"reaudit_soliman_{proto}_seed[0-9].csv")))
    dfs = [pd.read_csv(f).sort_values("record").reset_index(drop=True) for f in files]
    m = dfs[0][["record", "subject", "y", "null"]].copy()
    out = {}
    for model in ("cnngru", "nn1"):
        m[model] = np.mean([d[model].to_numpy() for d in dfs], 0)
        d, ci = cluster_ci(m.y.to_numpy(), m[model].to_numpy(), m["null"].to_numpy(), m.subject.astype(str).to_numpy(), rng, B=5000)
        out[model] = {"n_seeds": len(files), "mae": float(np.abs(m[model] - m.y).mean()), "diff_vs_null": d,
                      "diff_ci": [float(ci[0]), float(ci[1])]}
    out["null_mae"] = float(np.abs(m["null"] - m.y).mean())
    return out


def power(df_model, df_null, n_subj, per_subj, reps, rng, on="record"):
    m = df_model.merge(df_null[[on, "pred"]].rename(columns={"pred": "pred_null"}), on=on)
    m["subject"] = m.subject.astype(str)
    subs = m.subject.unique()
    groups = {s: g for s, g in m.groupby("subject")}
    hits, diffs = 0, []
    for _ in range(reps):
        pick = rng.choice(subs, n_subj, replace=False)
        parts = []
        for s in pick:
            g = groups[s]
            parts.append(g.sample(min(per_subj, len(g)), random_state=int(rng.integers(1 << 31))) if per_subj else g)
        sub = pd.concat(parts)
        d, ci = cluster_ci(sub.y.to_numpy(), sub.pred.to_numpy(), sub.pred_null.to_numpy(), sub.subject.to_numpy(), rng, B=1000)
        hits += ci[1] < 0; diffs.append(d)
    return {"n_subjects": n_subj, "per_subject": per_subj or "all", "reps": reps, "detection_rate": hits / reps,
            "mean_diff": float(np.mean(diffs))}


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    # 1. seed-ensemble decisions
    for tgt in ("sbp", "dbp"):
        files = sorted(glob.glob(os.path.join(RES, f"ppgbp_{tgt}_subject_kfold_1d_seed[0-9].csv")))
        if files:
            out[f"ppgbp_{tgt}_subject_kfold_1d_ensemble"] = ensemble_decision(
                files, os.path.join(RES, f"ppgbp_{tgt}_subject_kfold_mean_seed0.csv"), rng=rng)
    for proto in ("B", "C"):
        if glob.glob(os.path.join(RES, f"reaudit_soliman_{proto}_seed[0-9].csv")):
            out[f"reaudit_{proto}_ensemble"] = reaudit_ensemble(proto, rng)
    # 2. power at glucose-cohort sizes
    pw = {}
    for tgt in ("sbp", "dbp"):
        vm = os.path.join(RES, f"vitaldb_{tgt}_subject_kfold_1d_seed0.csv")
        vn = os.path.join(RES, f"vitaldb_{tgt}_subject_kfold_mean_seed0.csv")
        if os.path.exists(vm):
            dm, dn = pd.read_csv(vm), pd.read_csv(vn)
            pw[f"vitaldb_{tgt}_23x3"] = power(dm, dn, 23, 3, 500, rng)
            pw[f"vitaldb_{tgt}_16xall"] = power(dm, dn, 16, None, 500, rng)
        files = sorted(glob.glob(os.path.join(RES, f"ppgbp_{tgt}_subject_kfold_1d_seed[0-9].csv")))
        if files:
            dfs = [pd.read_csv(f) for f in files]
            dm = dfs[0].copy(); dm["pred"] = np.mean([d.pred.to_numpy() for d in dfs], 0)
            dn = pd.read_csv(os.path.join(RES, f"ppgbp_{tgt}_subject_kfold_mean_seed0.csv"))
            pw[f"ppgbp_{tgt}_23x1"] = power(dm, dn, 23, 1, 500, rng)
    out["power_at_glucose_sizes"] = pw
    json.dump(out, open(OUT, "w"), indent=2)
    for k, v in out.items():
        print(k, json.dumps(v)[:300])


if __name__ == "__main__":
    main()
