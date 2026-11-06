"""
make_supp_tables.py
===================
Generate the supplement tables that come straight from result files
(validation of the audit and the full PPG-BP grid) -> supp_tables.tex.

    python make_supp_tables.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "supp_tables.tex")


def _pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}"


def detector_table(v: dict) -> str:
    G = pd.DataFrame(v["grid"])
    C = G[G.frac_corrupt > 0]
    piv = {det: C.pivot_table(index="k", columns="p_rows", values=f"{det}_sensitivity") for det in ("global", "local")}
    ps = sorted(C.p_rows.unique())
    lines = [r"\begin{table}[h!]", r"\caption{Cross-join detector on synthetic label columns: subject-level sensitivity (\%)"
             r" by cycle length $k$ and fraction of a corrupted subject's rows that cycle, pooled over 10--100\,\% corrupted"
             r" subjects.  Specificity: " + f"{100 * G.global_specificity.min():.0f}\\,\\% (whole-column) and "
             f"{100 * G.local_specificity.min():.0f}\\,\\% (block scan) in every condition.}}",
             r"\label{tab:s_detector}", r"\centering", r"\begin{tabular}{@{}l" + "c" * (2 * len(ps)) + "@{}}", r"\toprule",
             r" & \multicolumn{%d}{c}{\textbf{Whole-column}} & \multicolumn{%d}{c}{\textbf{Block scan}} \\" % (len(ps), len(ps)),
             r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(ps), 2 + len(ps), 1 + 2 * len(ps)),
             r"$k$ & " + " & ".join([f"{int(p * 100)}\\,\\%" for p in ps] * 2) + r" \\", r"\midrule"]
    for k in piv["global"].index:
        cells = [_pct(piv[d].loc[k, p]) for d in ("global", "local") for p in ps]
        lines.append(f"{k} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    neg = v["real_negative_controls"]
    names = {"kermani_esmaeili": "Kermani--Esmaeili", "bigideas": "BIG~IDEAs", "stuba": "STUBA ePPG"}
    for key, r in neg.items():
        lines.append(rf"\multicolumn{{{1 + 2 * len(ps)}}}{{@{{}}l}}{{Real negative control, {names.get(key, key)}: "
                     rf"{r['n_flagged']} / {r['n_flagged_local']} of {r['n_subjects']} subjects flagged (whole-column / block)}} \\")
    pos = v.get("real_positive_control")
    if pos:
        lines.append(rf"\multicolumn{{{1 + 2 * len(ps)}}}{{@{{}}l}}{{Real positive control, Kaggle derivative: "
                     rf"{pos['n_flagged']} / {pos['n_flagged_local']} of {pos['n_with_more_than_one_label']} patients with "
                     rf"$>1$ reference value flagged}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def rule_table(rows: list[dict]) -> str:
    D = pd.DataFrame(rows)
    gains = sorted(D.signal_gain.unique())
    lines = [r"\begin{table}[h!]", r"\caption{Null-anchored decision rule on synthetic cohorts (%d per condition): percentage of"
             r" cohorts in which the model is declared better than the train-mean null (95\,\%% subject-cluster interval of"
             r" $\Delta$MAE entirely below zero), and the mean $\Delta$MAE (mg/dL).  $g$: true learnable signal in mg/dL per"
             r" bpm; at $g=0$ every claim is false (nominal one-sided rate 2.5\,\%%).}" % int(D.reps.iloc[0]),
             r"\label{tab:s_rule}", r"\centering", r"\begin{tabular}{@{}ll" + "c" * len(gains) + "@{}}", r"\toprule",
             r"\textbf{Protocol} & \textbf{Model} & " + " & ".join(f"$g={g:g}$" for g in gains) + r" \\", r"\midrule"]
    for pr, prn in (("subject_kfold", "subject-wise (honest)"), ("window_random", "random windows (leaky)")):
        for mo, mon in (("ridge", "ridge"), ("gbm", "GBM")):
            sub = D[(D.protocol == pr) & (D.model == mo)].set_index("signal_gain")
            cells = [f"{100 * sub.loc[g, 'rejection_rate']:.1f}\\,\\% ({sub.loc[g, 'mean_diff']:+.1f})" for g in gains]
            lines.append(f"{prn} & {mon} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def ppgbp_table(S: dict) -> str:
    runs = defaultdict(list)
    for m in S.values():
        runs[(m["target"], m["protocol"], m["model"])].append(m)
    models = [("mean", "Train mean (null)"), ("demo", "Ridge, demographics"), ("ridge", "Ridge, PPG features"),
              ("ridge+demo", "Ridge, PPG + demogr."), ("gbm+demo", "GBM, PPG + demogr."), ("1d", "1-D CNN")]
    fmt = lambda xs, d: (f"{np.mean(xs):.{d}f}" + (f"$\\pm${np.std(xs):.{d}f}" if len(xs) > 1 else "")) \
        if all(x is not None for x in xs) else "--"
    lines = [r"\begin{table}[h!]", r"\caption{PPG-BP, all rows.  MAE and RMSE in mmHg; networks mean\,$\pm$\,std over seeds;"
             r" $\Delta$: paired MAE difference against the train-mean null with its 95\,\% subject-cluster interval"
             r" (most conservative interval over seeds).}", r"\label{tab:s_ppgbp}", r"\centering", r"\small",
             r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
             r"\textbf{Target} & \textbf{Split} & \textbf{Model} & \textbf{MAE} & \textbf{RMSE} & $r$ & $\Delta$ [95\% CI] \\",
             r"\midrule"]
    for tgt in ("sbp", "dbp"):
        for pr, prn in (("segment_random", "random segments"), ("subject_kfold", "subject-wise")):
            for mo, mon in models:
                rs = runs.get((tgt, pr, mo))
                if not rs:
                    continue
                if "diff_ci" in rs[0]:
                    lo = max(r["diff_ci"][0] for r in rs); hi = max(r["diff_ci"][1] for r in rs)
                    dcell = f"{np.mean([r['diff_vs_null'] for r in rs]):+.1f} [{lo:+.2f}, {hi:+.2f}]"
                else:
                    dcell = "--"
                lines.append(f"{tgt.upper()} & {prn} & {mon} & {fmt([r['mae'] for r in rs], 1)} & "
                             f"{fmt([r['rmse'] for r in rs], 1)} & {fmt([r['pearson_r'] for r in rs], 2)} & {dcell} \\\\")
            lines.append(r"\midrule" if not (tgt == "dbp" and pr == "subject_kfold") else "")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(l for l in lines if l)


def main() -> None:
    parts = ["% Auto-generated by make_supp_tables.py -- do not edit by hand."]
    v = json.load(open(os.path.join(RES, "validation_detectors.json")))
    parts.append(r"\newcommand{\suppDetectorTable}{" + "\n" + detector_table(v["label_construction"]) + "\n}")
    parts.append(r"\newcommand{\suppRuleTable}{" + "\n" + rule_table(v["decision_rule"]) + "\n}")
    parts.append(r"\newcommand{\suppPpgbpTable}{" + "\n" + ppgbp_table(json.load(open(os.path.join(RES, "ppgbp_summary.json")))) + "\n}")
    open(OUT, "w").write("\n\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
