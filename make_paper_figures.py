"""
make_paper_figures.py
=====================
Build the paper figures from the out-of-fold prediction files in results/.

    python make_paper_figures.py            # writes figures/fig_*.pdf and .png

Figures
-------
fig_mendeley_loso      Clarke error grid of LOSO recording-level predictions (fusion)
                       + MAE of every model (mean / ridge / demographics / 1D / 2D / 3D / fusion)
fig_bigideas_tracking  CGM reference vs. prediction over the held-out period of one subject
                       + per-subject MAE: model vs. that subject's own train-period mean
fig_bigideas_clarke    Clarke error grid of pooled within-subject CGM-level predictions
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_clarke_error_grid import summarize_clarke_zones

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
# categorical slots 1-4 of the validated palette (blue, orange, aqua, yellow)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABELS = {"mean": "Train mean", "ridge": "Ridge (PPG features)", "demo": "Ridge (demographics)",
          "ridge+demo": "Ridge (PPG + demo.)", "1d": "1-D CNN", "2d": "2-D CNN (CWT)", "3d": "PointNet (Takens)",
          "fusion": "1D+2D+3D fusion"}
ORDER = ["mean", "demo", "ridge", "ridge+demo", "1d", "2d", "3d", "fusion"]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _clarke_boundaries(ax, hi):
    """Standard Clarke (1987) zone boundaries, clipped to [0, hi]."""
    segs = [([0, hi], [0, hi]), ([0, 175 / 3], [70, 70]), ([175 / 3, hi / 1.2], [70, hi]), ([70, 70], [84, hi]),
            ([0, 70], [180, 180]), ([70, 70 + (hi - 180)], [180, hi]), ([70, 70], [0, 56]),
            ([70, hi], [56, 0.8 * hi]), ([180, 180], [0, 70]), ([180, hi], [70, 70]), ([240, 240], [70, 180]),
            ([240, hi], [180, 180]), ([130, 180], [0, 70])]
    for i, (x, y) in enumerate(segs):
        ax.plot(x, y, color=MUTED, lw=0.6, ls=":" if i == 0 else "-", zorder=1)
    for z, (x, y) in {"A": (40, 22), "B": (140, 230), "B ": (235, 150), "C": (100, 240), "C ": (165, 20),
                      "D": (30, 120), "D ": (245, 100), "E": (30, 220), "E ": (220, 30)}.items():
        if x < hi and y < hi:
            ax.text(x, y, z.strip(), color=MUTED, fontsize=8, ha="center", va="center", zorder=2)


def _clarke(ax, y, p, title):
    hi = max(250.0, float(max(y.max(), p.max())) + 10)
    _clarke_boundaries(ax, hi)
    ax.scatter(y, p, s=9, c=C[0], alpha=0.75, linewidths=0.4, edgecolors="white", zorder=3)
    z = summarize_clarke_zones(y.astype(np.float32), p.astype(np.float32))
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    ax.set_xlabel("Reference glucose (mg/dL)"); ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_title(title, loc="left", color=INK)
    ax.text(0.98, 0.02, "  ".join(f"{k}: {z[f'zone_{k}_pct']:.1f}%" for k in "ABCDE" if z[f"zone_{k}_pct"] > 0),
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7, color=INK,
            bbox={"fc": "white", "ec": "none", "pad": 1.5})
    return z


def _load_summary(name):
    path = os.path.join(RES, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def _aggregate(summary, prefix=""):
    """Mean +/- std over seeds for each model tag '<prefix><model>_seed<k>'."""
    rows = {}
    for tag, m in summary.items():
        if not tag.startswith(prefix) or "mae" not in m or m.get("mae") is None:
            continue
        model = tag[len(prefix):].rsplit("_seed", 1)[0]
        rows.setdefault(model, []).append(m["mae"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in rows.items()}


def fig_mendeley():
    summary = _load_summary("subject_cv_summary.json")
    agg = _aggregate(summary)
    if not agg:
        print("no Mendeley results yet"); return
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.75})
    preds = sorted(glob.glob(os.path.join(RES, "subject_cv_fusion_seed*.csv")))
    if preds:
        df = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(preds)])
        df = df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))
        _clarke(a, df.y.to_numpy(), df.pred.to_numpy(),
                f"(a) LOSO, {len(df)} recordings, {df.subject.nunique()} subjects")
    models = [m for m in ORDER if m in agg]
    mae = [agg[m][0] for m in models]; err = [agg[m][1] for m in models]
    cols = [C[0] if m == "fusion" else (C[2] if m in ("1d", "2d", "3d") else "#b3b2ad") for m in models]
    b.barh(range(len(models)), mae, xerr=err, color=cols, height=0.62, error_kw={"lw": 0.8, "ecolor": INK})
    for i, v in enumerate(mae):
        b.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=7, color=INK)
    b.set_yticks(range(len(models))); b.set_yticklabels([LABELS[m] for m in models]); b.invert_yaxis()
    b.set_xlabel("Recording-level MAE (mg/dL), leave-one-subject-out"); b.grid(axis="x"); b.set_axisbelow(True)
    b.set_xlim(0, max(mae) * 1.18); b.set_title("(b) Model comparison", loc="left", color=INK)
    _save(fig, "fig_mendeley_loso")


def fig_bigideas(model="fusion"):
    summary = _load_summary("bigideas_summary.json")
    tag = None
    for model in (model, "1d", "2d", "3d"):            # fall back to the best available network
        tag = next((t for t in summary if t.startswith(f"within_{model}_seed")), None)
        if tag:
            break
    base = summary.get("within_mean_seed0")
    f = os.path.join(RES, f"bigideas_{tag}.csv") if tag else None
    if not (tag and base and os.path.exists(f)):
        print("no BIG IDEAs within-subject results yet"); return
    df = pd.read_csv(f).sort_values("record")
    # (a) tracking plot for the subject with most test readings
    sid = df.subject.astype(str).str.zfill(3).value_counts().index[0]
    g = df[df.subject.astype(str).str.zfill(3) == sid].reset_index(drop=True)
    hrs = np.arange(len(g)) * 5 / 60.0
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    a.plot(hrs, g.y, color=INK, lw=1.2, label="Dexcom G6 reference")
    a.plot(hrs, g.pred, color=C[0], lw=1.2, label=f"{LABELS[model]} (PPG)")
    a.set_xlabel(f"Held-out time, subject {sid} (h)"); a.set_ylabel("Glucose (mg/dL)"); a.grid(axis="y")
    a.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.5); a.set_title("(a) Within-subject tracking", loc="left", color=INK)
    # (b) per-subject MAE, model vs the subject's own train mean
    ps = summary[tag]["per_subject"]; pb = base["per_subject"]
    subs = sorted(ps)
    x = np.arange(len(subs)); w = 0.38
    b.bar(x - w / 2, [pb[s]["mae"] for s in subs], w, color="#b3b2ad", label="Subject train mean")
    b.bar(x + w / 2, [ps[s]["mae"] for s in subs], w, color=C[0], label=LABELS[model])
    b.set_xticks(x); b.set_xticklabels(subs, rotation=90); b.set_ylabel("MAE (mg/dL)"); b.grid(axis="y")
    b.set_axisbelow(True); b.set_title("(b) Per-subject error", loc="left", color=INK)
    b.set_ylim(0, max(max(pb[s]["mae"] for s in subs), max(ps[s]["mae"] for s in subs)) * 1.4)
    b.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.5)
    b.set_xlabel("Subject")
    _save(fig, "fig_bigideas_tracking")
    fig, a = plt.subplots(figsize=(3.4, 3.4))
    _clarke(a, df.y.to_numpy(), df.pred.to_numpy(), f"Within-subject, {len(df)} CGM readings")
    _save(fig, "fig_bigideas_clarke")


def fig_label_artifact():
    """Why clean-dataset.csv is unusable: labels cycle row by row (a); the raw
    recordings carry one label each, with real within-subject spread (b)."""
    csv = os.path.join(ROOT, "Archive_Dataset", "clean-dataset.csv")
    if not os.path.exists(csv):
        print("clean-dataset.csv absent; skipping artefact figure"); return
    from ppg_data import MENDELEY_ROOT
    df = pd.read_csv(csv, usecols=["Patient_Id", "Glucose_level", "PPG_Signal"], nrows=5000)
    p = df[df.Patient_Id == df.Patient_Id.iloc[0]].head(28)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a.step(range(len(p)), p.Glucose_level, where="mid", color=C[1], lw=1.4)
    a.scatter(range(len(p)), p.Glucose_level, s=10, color=C[1], zorder=3)
    a.set_xlabel("Row index in clean-dataset.csv (subject 1)"); a.set_ylabel("Glucose label (mg/dL)")
    a.set_title("(a) Derivative CSV: label cycles every row", loc="left", color=INK); a.grid(axis="y")
    a.set_title(f"(a) Derivative CSV: label cycles every row\n(PPG column = {int(p.PPG_Signal.iloc[0])} throughout)",
                loc="left", color=INK, fontsize=7.5)
    labs = sorted(glob.glob(os.path.join(MENDELEY_ROOT, "Labels", "label_01_*.csv")))
    g = [float(pd.read_csv(f).iloc[0]["Glucose"]) for f in labs]
    b.bar(range(len(g)), g, color=C[0], width=0.6)
    for i, v in enumerate(g):
        b.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=7, color=INK)
    b.set_xticks(range(len(g))); b.set_xticklabels([f"R{i + 1}" for i in range(len(g))])
    b.set_xlabel("Raw 10-s recordings, subject 01"); b.set_ylabel("Reference glucose (mg/dL)")
    b.set_ylim(0, max(g) * 1.2); b.set_title("(b) Raw data: one label per recording", loc="left", color=INK)
    _save(fig, "fig_label_artifact")


def fig_leakage_ladder():
    """MAE of the same models under progressively less leaky protocols."""
    path = os.path.join(RES, "leakage_ladder.json")
    if not os.path.exists(path):
        print("no leakage ladder yet"); return
    L = json.load(open(path))
    agg = {}
    for key, v in L.items():
        ds, proto, model, _ = key.split("/")
        agg.setdefault((ds, proto, model), []).append(v["mae"])
    # The subject-wise rung IS the leave-one-subject-out experiment of Table III: plot all its seeds,
    # exactly as make_results_macros.py does, so the figure and the table show one number per experiment.
    summ = _load_summary("subject_cv_summary.json")
    for model in ("1d", "fusion", "mean"):
        runs = [m["mae"] for t, m in summ.items() if t.rsplit("_seed", 1)[0] == model]
        if runs:
            agg[("mendeley", "subject", model)] = runs
    if ("mendeley", "subject", "mean") in agg:
        agg[("mendeley", "subject", "subjmean")] = agg[("mendeley", "subject", "mean")]
    n_k = len({str(r).zfill(2) for r in pd.read_csv(os.path.join(RES, "subject_cv_mean_seed0.csv")).subject})
    big = _load_summary("bigideas_summary.json").get("within_mean_seed0", {})
    n_b = len(big.get("subjects", [])) or 16
    panels = [("mendeley", ["window", "record", "subject"], ["random\nwindows", "random\nrecordings", "subject-\nwise"],
               f"(a) Kermani–Esmaeili, {n_k} subjects"),
              ("bigideas", ["reading", "chrono0", "chrono"], ["random\nreadings", "chrono-\nlogical", "chrono. +\n1-h gap"],
               f"(b) BIG IDEAs, {n_b} subjects")]
    models = [("subjmean", "Subject mean (leaky)", "#b3b2ad"), ("mean", "Null model", "#52514e"),
              ("1d", "1-D CNN", C[0]), ("fusion", "Fusion", C[1])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (ds, protos, labels, title) in zip(axes, panels):
        present = [m for m in models if any((ds, p, m[0]) in agg for p in protos)
                   and not (ds == "bigideas" and m[0] == "subjmean")]      # on BIG IDEAs the null IS the subject mean
        w = 0.8 / max(1, len(present))
        for j, (m, lab, col) in enumerate(present):
            xs = np.arange(len(protos)) + (j - (len(present) - 1) / 2) * w
            vals = [np.mean(agg.get((ds, p, m), [np.nan])) for p in protos]
            errs = [np.std(agg.get((ds, p, m), [0])) for p in protos]
            ax.bar(xs, vals, w * 0.92, yerr=errs, color=col, label=lab, error_kw={"lw": 0.7, "ecolor": INK})
            for x, v, e in zip(xs, vals, errs):
                if np.isfinite(v):
                    ax.text(x, v + e + 0.3, f"{v:.1f}", ha="center", fontsize=5, color=INK)
        ax.set_xticks(range(len(protos))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("MAE (mg/dL)"); ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    _save(fig, "fig_leakage_ladder")


def fig_validation():
    """(a) cross-join detector sensitivity, whole-column vs block scan; (b) decision-rule
    rejection rate vs true signal, honest vs leaky protocol, linear vs flexible model."""
    path = os.path.join(RES, "validation_detectors.json")
    if not os.path.exists(path):
        print("no validation yet"); return
    v = json.load(open(path))
    if "label_construction" not in v or "decision_rule" not in v:
        print("validation incomplete"); return
    G = pd.DataFrame(v["label_construction"]["grid"]); G = G[G.frac_corrupt > 0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.35})
    ks = sorted(G.k.unique()); ps = sorted(G.p_rows.unique())
    w = 0.13
    for j, (det, col, ls) in enumerate((("global", "#b3b2ad", "whole-column"), ("local", C[0], "block scan"))):
        for i, p in enumerate(ps):
            vals = [G[(G.k == k) & (G.p_rows == p)][f"{det}_sensitivity"].mean() * 100 for k in ks]
            x = np.arange(len(ks)) + (j * len(ps) + i - (2 * len(ps) - 1) / 2) * w
            a.bar(x, vals, w * 0.9, color=col, alpha=0.45 + 0.55 * (i + 1) / len(ps),
                  label=f"{ls}, {int(p * 100)}% rows" if True else None)
    a.set_xticks(range(len(ks))); a.set_xticklabels([f"k={k}" for k in ks])
    a.set_ylabel("Subjects detected (%)"); a.set_ylim(0, 150); a.set_yticks(range(0, 101, 20)); a.grid(axis="y"); a.set_axisbelow(True)
    a.set_title("(a) Cross-join detector (specificity 100%)", loc="left", color=INK)
    a.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    D = pd.DataFrame(v["decision_rule"])
    styles = {("subject_kfold", "ridge"): (C[0], "-", "honest, ridge"), ("subject_kfold", "gbm"): (C[0], "--", "honest, GBM"),
              ("window_random", "ridge"): (C[1], "-", "leaky, ridge"), ("window_random", "gbm"): (C[1], "--", "leaky, GBM")}
    for (pr, mo), (col, ls, lab) in styles.items():
        d = D[(D.protocol == pr) & (D.model == mo)].sort_values("signal_gain")
        b.plot(d.signal_gain, d.rejection_rate * 100, color=col, ls=ls, marker="o", ms=3.5, lw=1.4, label=lab)
    b.axhline(2.5, color=MUTED, lw=0.7, ls=":")
    b.text(D.signal_gain.max(), 4, "nominal 2.5%", ha="right", fontsize=6, color=MUTED)
    b.set_xlabel("True signal (mg/dL per bpm)"); b.set_ylabel("'Beats the null' (%)"); b.set_ylim(-3, 105)
    b.grid(axis="y"); b.set_axisbelow(True); b.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    b.set_title("(b) Decision rule: false claims and power", loc="left", color=INK)
    _save(fig, "fig_validation")


def fig_bp_ladder():
    """PPG-BP: the same models under the leaky segment split and the honest subject split."""
    path = os.path.join(RES, "ppgbp_summary.json")
    if not os.path.exists(path):
        print("no PPG-BP results yet"); return
    S = json.load(open(path))
    rows = {}
    for k, m in S.items():
        rows.setdefault((m["target"], m["protocol"], m["model"]), []).append(m)
    models = [("mean", "Null"), ("ridge", "Ridge\nPPG"), ("ridge+demo", "Ridge\nPPG+demo."), ("gbm+demo", "GBM\nPPG+demo."), ("1d", "1-D\nCNN")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"wspace": 0.3})
    for ax, (tgt, title) in zip(axes, (("sbp", "(a) Systolic BP"), ("dbp", "(b) Diastolic BP"))):
        x = np.arange(len(models)); w = 0.38
        for j, (pr, col, lab) in enumerate((("segment_random", C[1], "random segments (leaky)"),
                                            ("subject_kfold", C[0], "subject-wise (honest)"))):
            vals = [np.mean([r["mae"] for r in rows.get((tgt, pr, m), [{"mae": np.nan}])]) for m, _ in models]
            ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=col if True else None, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in models], fontsize=6)
        ax.set_ylabel("MAE (mmHg)"); ax.set_title(title, loc="left", color=INK); ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    _save(fig, "fig_bp_ladder")


def fig_leak_risk():
    """(a) synthetic: predicted a*P vs observed inflation and leaky gain; (b) real cohorts, log-log, retrospective vs
    preregistered prospective models."""
    path = os.path.join(RES, "leak_risk_eval.json")
    if not os.path.exists(path):
        print("no leak-risk evaluation"); return
    E = json.load(open(path)); L = json.load(open(os.path.join(RES, "leak_risk.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.32})
    sw = L.get("synthetic_sweep", [])
    px = [r["predicted_inflation"] for r in sw]
    a.scatter(px, [r["observed_inflation"] for r in sw], s=9, color=C[0], alpha=0.7, lw=0, label="observed inflation")
    a.scatter(px, [r["leaky_gain"] for r in sw], s=9, color="#b3b2ad", alpha=0.8, lw=0, label="leaky gain over null")
    lim = max(max(px), max(r["observed_inflation"] for r in sw)) * 1.05
    a.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    a.axhline(0, color=GRID, lw=0.8)
    a.set_xlabel("Predicted inflation $a\\,P$ (mg/dL)"); a.set_ylabel("Observed (mg/dL)")
    a.set_title("(a) Synthetic cohorts, gradient boosting", loc="left", color=INK)
    a.grid(True); a.set_axisbelow(True); a.legend(frameon=False, fontsize=6, loc="upper left")
    shape = {"demo": "D", "ppg+demo": "s", "ppg": "o", "raw": "^"}
    cols = {"vitaldb": C[0], "ppgbp": C[1]}
    for r in E["rows"]:
        if not r["flexible"] or r["observed"] <= 0 or r["predicted"] <= 0:
            continue
        coh = r["cohort"].split("_")[0]
        inp = L["cohorts"][r["cohort"]]["models"][r["model"]]["inputs"]
        kw = dict(marker=shape.get(inp, "o"), s=30, zorder=3)
        if r["prospective"]:
            b.scatter(r["predicted"], r["observed"], facecolor="white", edgecolor=cols[coh], lw=1.2, **kw)
        else:
            b.scatter(r["predicted"], r["observed"], color=cols[coh], edgecolor="white", lw=0.4, **kw)
    b.set_xscale("log"); b.set_yscale("log")
    lo, hi = 0.08, 25
    b.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    b.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2], color=GRID, alpha=0.6, lw=0, zorder=0)
    b.set_xlim(lo, hi); b.set_ylim(0.15, hi)
    b.set_xlabel("Predicted inflation $a\\,P$ (mmHg)"); b.set_ylabel("Observed inflation (mmHg)")
    b.set_title("(b) Blood-pressure cohorts, flexible models", loc="left", color=INK)
    b.grid(True, which="major"); b.set_axisbelow(True)
    g = "#6b6a66"
    h = [plt.Line2D([], [], marker="D", ls="", color=g, ms=4), plt.Line2D([], [], marker="s", ls="", color=g, ms=4),
         plt.Line2D([], [], marker="o", ls="", color=g, ms=4), plt.Line2D([], [], marker="^", ls="", color=g, ms=4),
         plt.Line2D([], [], marker="o", ls="", color=g, ms=5),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=g, ms=5),
         plt.Line2D([], [], marker="s", ls="", color=C[0], ms=5), plt.Line2D([], [], marker="s", ls="", color=C[1], ms=5)]
    l = ["demographics", "PPG + demographics", "PPG features", "raw waveform", "retrospective", "preregistered",
         "VitalDB", "PPG-BP"]
    b.legend(h, l, frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    _save(fig, "fig_leak_risk")


if __name__ == "__main__":
    fig_leak_risk()
    fig_validation()
    fig_bp_ladder()
    fig_label_artifact()
    fig_mendeley()
    fig_bigideas()
    fig_leakage_ladder()
