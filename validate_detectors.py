"""
validate_detectors.py
=====================
Methods-paper validation of the leakage audit, on data where the truth is
known by construction.

(A) Label-construction (cross-join) detector, ``ppgbench.integrity.detect_label_cycling``
    * Synthetic label columns: clean subjects carry record-structured labels
      (each record's label repeated over its windows, record labels drawn
      independently); corrupted subjects carry the cross-join artefact (the
      subject's k reference values cycled row by row).  Grid over k (2..10),
      corrupted fraction of subjects (0, 10, 25, 50, 100 %) and partial
      corruption (a fraction p of a corrupted subject's rows cycled, the rest
      clean).  Reported: subject-level sensitivity and specificity.
    * Real negative controls: the window-label columns of the three clean
      cohorts (Kermani-Esmaeili, BIG IDEAs, STUBA) -- any flag is a false alarm.
    * Real positive control: the Kaggle derivative (every patient should flag).

(B) Null-anchored decision rule ("a model beats the null iff the paired
    subject-cluster bootstrap interval of its MAE difference lies entirely
    below zero"), evaluated on synthetic cohorts (ppgbench ``synthetic``
    generator ``make_cohort``) with known signal strength g (mg/dL per bpm;
    g = 0 means no learnable signal) and a label-independent per-subject
    pulse-shape fingerprint, under the honest subject-wise protocol and under the
    leaky random-window protocol, for a linear (ridge) and a flexible
    (gradient-boosting) model.  Reported: rejection rate (false-positive rate
    at g = 0, power at g > 0) and the mean MAE difference.

    python validate_detectors.py --reps 60      # -> results/validation_detectors.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from ppgbench.integrity import detect_label_cycling, detect_label_cycling_local
from ppg_data import handcrafted_features

log = logging.getLogger("validate")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "validation_detectors.json")


# --------------------------------------------------------------------------- #
# (A) label-construction detector
# --------------------------------------------------------------------------- #
def synth_label_column(rng, n_subjects, k, frac_corrupt, p_rows, rows_per_record=200, n_records=None):
    """Concatenated label column + group ids + per-subject truth (True = corrupted)."""
    labels, groups, truth = [], [], {}
    corrupt = set(rng.choice(n_subjects, int(round(frac_corrupt * n_subjects)), replace=False).tolist())
    for s in range(n_subjects):
        vals = np.round(rng.normal(115, 18, size=k)).astype(float)       # this subject's k reference values
        clean = np.repeat(vals, rows_per_record)                           # record-structured runs
        if s in corrupt:
            col = clean.copy()
            m = rng.random(len(col)) < p_rows                              # rows replaced by the cross-join pattern
            idx = np.arange(len(col))
            col[m] = vals[idx[m] % k]
            # the artefact is contiguous in the real file: cycle a contiguous block instead of scattered rows
            n_block = int(round(p_rows * len(col)))
            col = clean.copy()
            col[:n_block] = vals[np.arange(n_block) % k]
        else:
            col = clean
        labels.append(col); groups.append(np.full(len(col), s)); truth[str(s)] = s in corrupt
    return np.concatenate(labels), np.concatenate(groups), truth


def part_a(reps, rng):
    out = {"grid": []}
    for k in (2, 3, 5, 7, 10):
        for frac in (0.0, 0.1, 0.25, 0.5, 1.0):
            for p in (0.25, 0.5, 1.0):
                if frac == 0.0 and p != 1.0:
                    continue
                cnt = {"global": [0, 0, 0, 0], "local": [0, 0, 0, 0]}          # tp, fn, tn, fp
                for _ in range(reps):
                    lab, grp, truth = synth_label_column(rng, 20, k, frac, p)
                    flags = {"global": set(detect_label_cycling(lab, groups=grp).get("groups_cycling", [])),
                             "local": set(detect_label_cycling_local(lab, groups=grp)["groups_cycling"])}
                    for det, flagged in flags.items():
                        c = cnt[det]
                        for g, bad in truth.items():
                            hit = g in flagged
                            c[0] += bad and hit; c[1] += bad and not hit; c[3] += (not bad) and hit; c[2] += (not bad) and not hit
                row = {"k": k, "frac_corrupt": frac, "p_rows": p}
                for det, (tp, fn, tn, fp) in cnt.items():
                    row[f"{det}_sensitivity"] = tp / (tp + fn) if tp + fn else None
                    row[f"{det}_specificity"] = tn / (tn + fp) if tn + fp else None
                out["grid"].append(row)
    # real negative controls: window-label columns in file order, grouped by subject
    from ppg_data import load_bigideas, load_mendeley, load_stuba
    neg = {}
    for name, ws in (("kermani_esmaeili", load_mendeley()), ("stuba", load_stuba()),
                     ("bigideas", load_bigideas(max_windows_per_egv=2))):
        res = detect_label_cycling(ws.y, groups=ws.subject)
        loc = detect_label_cycling_local(ws.y, groups=ws.subject)
        neg[name] = {"n_subjects": res["n_groups"], "n_flagged": res["n_groups_cycling"], "flagged": res["groups_cycling"],
                     "n_flagged_local": loc["n_groups_cycling"]}
    out["real_negative_controls"] = neg
    csv = os.path.join(ROOT, "Archive_Dataset", "clean-dataset.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, usecols=["Patient_Id", "Glucose_level"])
        res = detect_label_cycling(df.Glucose_level.to_numpy(), groups=df.Patient_Id.to_numpy())
        multi = int(sum(1 for v in res["per_group"].values() if v["n_unique"] > 1))
        loc = detect_label_cycling_local(df.Glucose_level.to_numpy(), groups=df.Patient_Id.to_numpy())
        out["real_positive_control"] = {"n_patients": res["n_groups"], "n_with_more_than_one_label": multi,
                                        "n_flagged": res["n_groups_cycling"], "n_flagged_local": loc["n_groups_cycling"]}
    return out


def part_a_stress(reps, rng):
    """Failure modes of the cross-join detector, reported as limitations.

    shuffled_crossjoin : a corrupted subject's cross-joined rows are randomly
        permuted (the label multiset is unchanged, the cycle is destroyed).
    window_major_genuine : a *genuine* table (one label per record) stored
        window-major, i.e. window 1 of every record, then window 2 of every
        record ...; its label column cycles with period = number of records.
    """
    out = {"shuffled_crossjoin": [], "window_major_genuine": []}
    for k in (3, 7, 10):
        hit_g = hit_l = 0
        for _ in range(reps):
            vals = np.round(rng.normal(115, 18, size=k))
            col = vals[np.arange(k * 200) % k]
            rng.shuffle(col)
            hit_g += bool(detect_label_cycling(col)["cycling"]); hit_l += bool(detect_label_cycling_local(col)["cycling"])
        out["shuffled_crossjoin"].append({"k": k, "global_sensitivity": hit_g / reps, "local_sensitivity": hit_l / reps})
    for n_rec in (3, 10, 30, 60):
        fa_g = fa_l = 0
        for _ in range(reps):
            labs = np.round(rng.normal(115, 18, size=n_rec))
            col = np.tile(labs, 40)                      # 40 windows per record, window-major order
            fa_g += bool(detect_label_cycling(col)["cycling"]); fa_l += bool(detect_label_cycling_local(col)["cycling"])
            # the same table in record order (what the audit requires) is clean
        rec_order = np.repeat(labs, 40)
        out["window_major_genuine"].append({"n_records": n_rec, "global_false_alarm": fa_g / reps,
                                            "local_false_alarm": fa_l / reps,
                                            "record_order_flagged": bool(detect_label_cycling_local(rec_order)["cycling"])})
    return out


# --------------------------------------------------------------------------- #
# (B) decision-rule calibration
# --------------------------------------------------------------------------- #
GAINS = (0.0, 1.0, 2.0, 3.0, 4.0)


def make_cohort(seed, gain, n_subjects=16, n_records=12, n_windows=3, between_sd=20.0, hr_within_sd=8.0,
                label_noise_sd=3.0):
    """Synthetic cohort with known ground truth.

    label = subject offset (unlearnable, SD ``between_sd``) + gain * (HR - 70) + noise.
    Each subject also has a pulse-shape fingerprint (harmonic and notch amplitude),
    unrelated to the label, so a flexible model *can* learn identity -- and hence
    the subject offset -- whenever a protocol lets a subject's windows into training.
    """
    from ppgbench.core.signal import bandpass, sliding_windows, zscore_rows
    from ppgbench.data.registry import FS, HOP, WIN
    from ppgbench.data.synthetic import synthetic_ppg
    rng = np.random.default_rng(seed)
    n = (n_windows - 1) * HOP + WIN
    X, y, subj, rec = [], [], [], []
    for s in range(n_subjects):
        offset = rng.normal(120.0, between_sd); hr_rest = rng.uniform(60.0, 85.0)
        harmonic = rng.uniform(0.15, 0.75); notch = rng.uniform(0.05, 0.45)
        for i in range(n_records):
            hr = float(np.clip(hr_rest + rng.normal(0.0, hr_within_sd), 40.0, 160.0))
            lab = offset + gain * (hr - 70.0) + rng.normal(0.0, label_noise_sd)
            W = sliding_windows(bandpass(synthetic_ppg(n, FS, hr, rng, harmonic=harmonic, notch=notch), FS), WIN, HOP)
            X.append(W); y += [lab] * len(W); subj += [s] * len(W); rec += [s * 1000 + i] * len(W)
    return zscore_rows(np.concatenate(X)).astype(np.float32), np.array(y), np.array(subj), np.array(rec)


def cluster_ci(y, p_model, p_null, subj, rng, B=2000):
    """Paired MAE difference and its 95 % subject-cluster bootstrap interval (vectorised)."""
    subs, inv = np.unique(subj, return_inverse=True)
    S = len(subs)
    em = np.bincount(inv, np.abs(p_model - y), S); en = np.bincount(inv, np.abs(p_null - y), S)
    nn = np.bincount(inv, minlength=S).astype(float)
    W = rng.multinomial(S, np.full(S, 1.0 / S), size=B).astype(float)
    d = (W @ em - W @ en) / (W @ nn)
    return float((em.sum() - en.sum()) / nn.sum()), np.percentile(d, [2.5, 97.5])


def fit(model, F, y, tr, te):
    if model == "gbm":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=5,
                                             random_state=0).fit(F[tr], y[tr]).predict(F[te])
    sc = StandardScaler().fit(F[tr])
    return RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(F[tr]), y[tr]).predict(sc.transform(F[te]))


def part_b(reps, rng, gains=GAINS):
    acc = {}
    for gain in gains:
        t0 = time.time()
        for rep in range(reps):
            X, y, subj, rec = make_cohort(100_000 * int(round(gain * 10)) + rep, gain)
            F = handcrafted_features(X)
            for protocol in ("subject_kfold", "window_random"):
                folds = (list(GroupKFold(4).split(F, y, subj)) if protocol == "subject_kfold"
                         else list(KFold(4, shuffle=True, random_state=rep).split(F)))
                for model in ("ridge", "gbm"):
                    pm = np.empty(len(y)); pn = np.empty(len(y))
                    for tr, te in folds:
                        pm[te] = fit(model, F, y, tr, te); pn[te] = y[tr].mean()
                    r = (pd.DataFrame({"rec": rec, "subj": subj, "y": y, "pm": pm, "pn": pn})
                         .groupby("rec", as_index=False).agg(subj=("subj", "first"), y=("y", "first"),
                                                             pm=("pm", "mean"), pn=("pn", "mean")))
                    d, ci = cluster_ci(r.y.to_numpy(), r.pm.to_numpy(), r.pn.to_numpy(), r.subj.to_numpy(), rng)
                    a = acc.setdefault((gain, protocol, model), {"rej": 0, "d": []})
                    a["rej"] += int(ci[1] < 0); a["d"].append(d)
        for protocol in ("subject_kfold", "window_random"):
            for model in ("ridge", "gbm"):
                a = acc[(gain, protocol, model)]
                log.info("B gain %.1f %-14s %-5s claims %.3f  mean diff %+.2f", gain, protocol, model,
                         a["rej"] / reps, np.mean(a["d"]))
        log.info("gain %.1f done in %.0f s", gain, time.time() - t0)
    return [{"signal_gain": g, "protocol": p, "model": m, "reps": reps, "rejection_rate": a["rej"] / reps,
             "mean_diff": float(np.mean(a["d"])), "sd_diff": float(np.std(a["d"]))} for (g, p, m), a in acc.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--reps-a", type=int, default=50)
    ap.add_argument("--parts", nargs="+", default=["a", "b"], help="a, b, stress, null (g=0 only, --reps cohorts)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ppgbench.integrity").setLevel(logging.ERROR)
    rng = np.random.default_rng(0)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    if "a" in args.parts:
        out["label_construction"] = part_a(args.reps_a, rng)
        json.dump(out, open(OUT, "w"), indent=2)
        log.info("part A done in %.0f s", time.time() - t0)
    if "stress" in args.parts:
        out["detector_stress"] = part_a_stress(args.reps_a, rng)
        json.dump(out, open(OUT, "w"), indent=2)
        log.info("stress done: %s", json.dumps(out["detector_stress"]))
    if "null" in args.parts:
        out["decision_rule_null"] = part_b(args.reps, rng, gains=(0.0,))
        json.dump(out, open(OUT, "w"), indent=2)
        log.info("null-only calibration done in %.0f s", time.time() - t0)
    if "b" in args.parts:
        out["decision_rule"] = part_b(args.reps, rng)
        json.dump(out, open(OUT, "w"), indent=2)
        log.info("part B done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
