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
    models = [("mean", "Train mean (null)"), ("nn1", "1-NN copy probe"), ("demo", "Ridge, demographics"),
              ("ridge", "Ridge, PPG features"), ("ridge+demo", "Ridge, PPG + demogr."), ("gbm", "GBM, PPG features"),
              ("gbm+demo", "GBM, PPG + demogr."), ("1d", "1-D CNN")]
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


def reaudit_table(R: dict) -> str:
    PN = {"A": "A: augment, then split samples", "B": "B: split recordings, augment train",
          "C": "C: split subjects, augment train"}
    MN = {"null": "Null", "nn1": "1-NN copy probe", "cnngru": "CNN--GRU"}
    lines = [r"\begin{table}[h!]", r"\caption{Re-audit of Soliman \emph{et al.} (arXiv:2411.11094), every seed.  MAE and RMSE in"
             r" mg/dL; $\Delta$: paired MAE difference against the null with its 95\,\% subject-cluster interval; A: Clarke"
             r" zone A (\%).  Stated elements reproduced: 0.5--8 Hz band-pass, 30 Hz, Gaussian-noise augmentation (67 to 268"
             r" samples), shuffled 10-fold CV.  Unstated elements resolved as described in the main text.}",
             r"\label{tab:s_reaudit}", r"\centering", r"\small", r"\begin{tabular}{@{}lllccccc@{}}", r"\toprule",
             r"\textbf{Protocol} & \textbf{Seed} & \textbf{Model} & \textbf{MAE} & \textbf{RMSE} & $R^2$ & \textbf{A} & $\Delta$ [95\% CI] \\",
             r"\midrule"]
    for pr, prn in PN.items():
        import re as _re
        for k in sorted(x for x in R if x.startswith(pr + "_seed")):
            m_ = _re.fullmatch(pr + r"_seed(\d+)(?:_noise([\d.]+))?", k)
            seed_lab = m_.group(1) + (f", noise $\\times${m_.group(2)}" if m_.group(2) else "")
            for mo, mon in MN.items():
                r = R[k][mo]
                d = f"{r['diff_vs_null']:+.2f} [{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}]" if "diff_ci" in r else "--"
                lines.append(f"{prn} & {seed_lab} & {mon} & {r['mae']:.2f} & {r['rmse']:.2f} & {r['r2']:.2f} & "
                             f"{r['zone_A_pct']:.0f} & {d} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def vitaldb_table(S: dict) -> str:
    runs = defaultdict(list)
    for m in S.values():
        runs[(m["target"], m["protocol"], m["model"])].append(m)
    models = [("mean", "Null (train mean; case mean if personal)"), ("nn1", "1-NN copy probe"), ("demo", "Ridge, demographics"),
              ("ridge", "Ridge, PPG features"), ("ridge+demo", "Ridge, PPG + demogr."), ("gbm", "GBM, PPG features"),
              ("gbm+demo", "GBM, PPG + demogr."), ("1d", "1-D CNN")]
    PR = [("segment_random", "random segments"), ("subject_kfold", "subject-wise"), ("personal", "personal, absolute"),
          ("personal_res", "personal, residual")]
    f = lambda xs, d: f"{np.mean(xs):.{d}f}" if all(x is not None for x in xs) else "--"
    lines = [r"\begin{table}[h!]", r"\caption{VitalDB arterial-line blood pressure, all rows.  MAE, mean error (ME) and SD of"
             r" error in mmHg (AAMI/ISO 81060-2 limits: $|$ME$|\le5$, SD $\le8$); $r$: pooled correlation; $r_w$: median"
             r" within-case correlation; $\Delta$: paired MAE difference against the protocol's null with its 95\,\% patient-cluster"
             r" interval.}", r"\label{tab:s_vitaldb}", r"\centering", r"\scriptsize",
             r"\begin{tabular}{@{}lllcccccc@{}}", r"\toprule",
             r"\textbf{Target} & \textbf{Split} & \textbf{Model} & \textbf{MAE} & \textbf{ME} & \textbf{SD} & $r$ & $r_w$ & $\Delta$ [95\% CI] \\",
             r"\midrule"]
    for tgt in ("sbp", "dbp"):
        for pr, prn in PR:
            any_row = False
            for mo, mon in models:
                rs = runs.get((tgt, pr, mo))
                if not rs:
                    continue
                any_row = True
                d = (f"{np.mean([r['diff_vs_null'] for r in rs]):+.2f} [{min(r['diff_ci'][0] for r in rs):+.2f}, "
                     f"{max(r['diff_ci'][1] for r in rs):+.2f}]") if "diff_ci" in rs[0] else "--"
                lines.append(f"{tgt.upper()} & {prn} & {mon} & {f([r['mae'] for r in rs], 1)} & {f([r['me'] for r in rs], 1)} & "
                             f"{f([r['sde'] for r in rs], 1)} & {f([r.get('pearson_r') for r in rs], 2)} & "
                             f"{f([r.get('within_case_r_median') for r in rs], 2)} & {d} \\\\")
            if any_row:
                lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def review_table(v: dict, st: dict, cp: dict) -> str:
    """Detector stress tests, copy-probe validation, null calibration, power at glucose sizes, seed ensembles."""
    L = [r"\begin{table}[h!]", r"\caption{Additional validation.  Detector stress tests (100 draws each); copy probe on the"
         r" Kermani--Esmaeili ladder; decision-rule false-claim rate with no signal (exact binomial 95\,\% CI); detection rate"
         r" of the blood-pressure signal when the honest predictions are subsampled to glucose-cohort sizes (500 draws; models"
         r" trained on the full cohort); seed-ensemble decisions.}", r"\label{tab:s_review}", r"\centering", r"\small",
         r"\begin{tabular}{@{}p{7.2cm}p{6.8cm}@{}}", r"\toprule", r"\textbf{Analysis} & \textbf{Result} \\", r"\midrule"]
    ds = v.get("detector_stress", {})
    for r in ds.get("shuffled_crossjoin", []):
        L.append(f"Shuffled cross-join, $k={r['k']}$ & sensitivity {100*r['global_sensitivity']:.0f}\\,\\% (whole column), "
                 f"{100*r['local_sensitivity']:.0f}\\,\\% (block scan) \\\\")
    for r in ds.get("window_major_genuine", []):
        L.append(f"Genuine table stored window-major, {r['n_records']} records & false alarm {100*r['global_false_alarm']:.0f}\\,\\% / "
                 f"{100*r['local_false_alarm']:.0f}\\,\\%; record order: {'flagged' if r['record_order_flagged'] else 'clean'} \\\\")
    L.append(r"\midrule")
    names = {"window_random": "random windows", "record_random": "random recordings", "subject": "subject-wise"}
    for k, r in cp.items():
        rung, kind = k.rsplit("_", 1)
        L.append(f"Copy probe ({'Euclidean' if kind == 'euclid' else 'shift-invariant'}), {names[rung]} & $\\Delta$ "
                 f"{r['diff_vs_null']:+.2f} [{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}] mg/dL \\\\")
    L.append(r"\midrule")
    from scipy.stats import beta
    for r in v.get("decision_rule_null", []):
        k = int(round(r["rejection_rate"] * r["reps"])); n = int(r["reps"])
        lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        L.append(f"No signal, {'subject-wise' if r['protocol'] == 'subject_kfold' else 'random windows'}, {r['model']} ({n} cohorts) & "
                 f"false claims {100*r['rejection_rate']:.1f}\\,\\% [{100*lo:.1f}, {100*hi:.1f}] \\\\")
    L.append(r"\midrule")
    lab = {"vitaldb_sbp_23x3": "VitalDB SBP, 23 patients $\\times$ 3 segments", "vitaldb_sbp_16xall": "VitalDB SBP, 16 patients $\\times$ all",
           "ppgbp_sbp_23x1": "PPG-BP SBP, 23 subjects", "vitaldb_dbp_23x3": "VitalDB DBP, 23 patients $\\times$ 3 segments",
           "vitaldb_dbp_16xall": "VitalDB DBP, 16 patients $\\times$ all", "ppgbp_dbp_23x1": "PPG-BP DBP, 23 subjects"}
    for k, r in st.get("power_at_glucose_sizes", {}).items():
        L.append(f"{lab.get(k, k)} (1-D CNN) & detected in {100*r['detection_rate']:.0f}\\,\\% of draws (mean $\\Delta$ {r['mean_diff']:+.2f}) \\\\")
    L.append(r"\midrule")
    for k, r in st.items():
        if k.endswith("_ensemble") and k.startswith("ppgbp"):
            L.append(f"PPG-BP {k.split('_')[1].upper()} 1-D CNN, seed ensemble of {r['n_seeds']} & MAE {r['mae']:.2f}, $\\Delta$ "
                     f"{r['diff_vs_null']:+.2f} [{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}] \\\\")
        if k.startswith("reaudit_") and k.endswith("_ensemble"):
            c = r["cnngru"]
            L.append(f"Re-audit protocol {k.split('_')[1]}, CNN--GRU seed ensemble of {c['n_seeds']} & MAE {c['mae']:.2f} (null "
                     f"{r['null_mae']:.2f}), $\\Delta$ {c['diff_vs_null']:+.2f} [{c['diff_ci'][0]:+.2f}, {c['diff_ci'][1]:+.2f}] \\\\")
    L[-1] = L[-1]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def leak_risk_table(L: dict) -> str:
    MN = {"ridge": "Ridge, PPG", "gbm": "GBM, PPG", "1d": "1-D CNN", "demo": "Ridge, demo.", "ridge+demo": "Ridge, PPG+demo.",
          "gbm+demo": "GBM, PPG+demo.", "gbm_demo": "GBM, demo. only", "rf_demo": "RF, demo. only$^\\dagger$",
          "knn_demo": "5-NN, demo. only$^\\dagger$", "rf+demo": "RF, PPG+demo.$^\\dagger$", "et": "Extra-trees, PPG$^\\dagger$",
          "gbm+demo_l50": "GBM (leaf 50), PPG+demo.$^\\dagger$"}
    CN = {"vitaldb_sbp": "VitalDB SBP", "vitaldb_dbp": "VitalDB DBP", "ppgbp_sbp": "PPG-BP SBP", "ppgbp_dbp": "PPG-BP DBP"}
    T = [r"\begin{table}[h!]", r"\caption{A-priori leak-risk prediction on the blood-pressure cohorts.  ICC: intraclass correlation"
         r" of the labels by subject; $P$: potential (mmHg); $a$: 1-NN subject identifiability under the random-segment split from"
         r" PPG features, demographics, or both; predicted inflation $\hat I=a\,P$ for the model's inputs; observed inflation ="
         r" subject-wise MAE minus random-segment MAE.  $^\dagger$prospective: predictions committed (commit 47d6ae2)"
         r" before the model was run.}", r"\label{tab:s_leakrisk}", r"\centering", r"\small",
         r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"\textbf{Cohort} & \textbf{Summary / model} & \textbf{Inputs} & $a$ & $\hat I$ & \textbf{Observed} \\", r"\midrule"]
    for key, cn in CN.items():
        d = L.get("cohorts", {}).get(key)
        if not d:
            continue
        T.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{cn}}}: ICC {d['icc']:.2f}, $P$ {d['potential']:.2f}, $a$: PPG {100*d['identifiability']['ppg']:.0f}\\,\\%, "
                 f"demo.\\ {100*d['identifiability']['demo']:.0f}\\,\\%, both {100*d['identifiability']['ppg+demo']:.0f}\\,\\%, raw "
                 f"{100*d['identifiability']['raw']:.0f}\\,\\%" + (f", $P_{{time}}$ {d['potential_temporal']:.2f}" if "potential_temporal" in d else "")
                 + "} \\\\")
        for m, r in d.get("models", {}).items():
            T.append(f" & {MN.get(m, m)} & {r['inputs'].replace('+', ' + ')} & {100*r['identifiability']:.0f}\\,\\% & "
                     f"{r['predicted_inflation']:.2f} & {r['inflation']:+.2f} \\\\")
        T.append(r"\midrule")
    T[-1] = r"\bottomrule"
    T += [r"\end{tabular}", r"\end{table}"]
    sw = L.get("synthetic_sweep", [])
    if sw:
        df = pd.DataFrame(sw).groupby(["between_sd", "fp_scale"], as_index=False).mean(numeric_only=True)
        T += [r"\begin{table}[h!]", r"\caption{Synthetic leak-risk sweep (8 cohorts per cell; gradient boosting): mean identifiability,"
              r" potential, predicted and observed inflation (mg/dL) by between-subject label SD and fingerprint scale.}",
              r"\label{tab:s_leaksweep}", r"\centering", r"\small", r"\begin{tabular}{@{}cccccccc@{}}", r"\toprule",
              r"\textbf{Between-subject SD} & \textbf{Fingerprint} & $a$ & $P$ & $\hat I$ & \textbf{Observed} & \textbf{Leaky gain} & \textbf{Honest gain} \\", r"\midrule"]
        for _, r in df.iterrows():
            T.append(f"{r.between_sd:.0f} & {r.fp_scale:.2f} & {100*r.identifiability:.0f}\\,\\% & {r.potential:.2f} & "
                     f"{r.predicted_inflation:.2f} & {r.observed_inflation:+.2f} & {r.leaky_gain:+.2f} & {r.honest_gain:+.2f} \\\\")
        T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(T)


def main() -> None:
    parts = ["% Auto-generated by make_supp_tables.py -- do not edit by hand."]
    v = json.load(open(os.path.join(RES, "validation_detectors.json")))
    parts.append(r"\newcommand{\suppDetectorTable}{" + "\n" + detector_table(v["label_construction"]) + "\n}")
    parts.append(r"\newcommand{\suppRuleTable}{" + "\n" + rule_table(v["decision_rule"]) + "\n}")
    parts.append(r"\newcommand{\suppPpgbpTable}{" + "\n" + ppgbp_table(json.load(open(os.path.join(RES, "ppgbp_summary.json")))) + "\n}")
    p = os.path.join(RES, "reaudit_soliman.json")
    if os.path.exists(p):
        parts.append(r"\newcommand{\suppReauditTable}{" + "\n" + reaudit_table(json.load(open(p))) + "\n}")
    p = os.path.join(RES, "vitaldb_summary.json")
    if os.path.exists(p):
        parts.append(r"\newcommand{\suppVitaldbTable}{" + "\n" + vitaldb_table(json.load(open(p))) + "\n}")
    st_p, cp_p = os.path.join(RES, "stats_tbme.json"), os.path.join(RES, "copy_probe.json")
    st = json.load(open(st_p)) if os.path.exists(st_p) else {}
    cp = json.load(open(cp_p)) if os.path.exists(cp_p) else {}
    parts.append(r"\newcommand{\suppReviewTable}{" + "\n" + review_table(v, st, cp) + "\n}")
    lr_p = os.path.join(RES, "leak_risk.json")
    if os.path.exists(lr_p):
        parts.append(r"\newcommand{\suppLeakRiskTable}{" + "\n" + leak_risk_table(json.load(open(lr_p))) + "\n}")
    open(OUT, "w").write("\n\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
