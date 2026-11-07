"""
leak_risk.py
============
A-priori prediction of identity-leak inflation, computable before any target
model is trained.

Under a split that lets a subject's other samples into training (random
segments), a model can lower its error by recognising the subject and
recalling the subject's label level.  Two quantities bound how much:

    potential  P = MAE(train-mean null) - MAE(subject oracle)
               where the subject oracle predicts each test sample by the mean
               label of the same subject's *training* samples (falling back to
               the global mean).  P depends only on the labels and the split:
               it is the error an identity lookup removes when identification
               is perfect.  It grows with the between-subject share of label
               variance (intraclass correlation).
    identifiability  a(X) = accuracy of 1-nearest-neighbour subject
               identification from the inputs X (standardised, per sample)
               under the same split -- how often the inputs reveal *who* the
               test sample belongs to.

A model that identified subjects as well as 1-NN and then recalled their
level would score  MAE_leak ~ a * MAE_oracle + (1 - a) * MAE_null,  i.e. an
identity-driven gain of  a * P.  The predicted inflation (honest MAE minus
leaky MAE) is therefore  I_hat(X) = a(X) * P.  It is compared with the
inflation observed for each model on each cohort.

    python leak_risk.py          # -> results/leak_risk.json
"""
from __future__ import annotations

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ppg_data import handcrafted_features

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "leak_risk.json")


def icc1(y, g):
    """One-way random-effects ICC(1) of labels y grouped by g."""
    df = pd.DataFrame({"y": y, "g": g})
    k = df.groupby("g").size()
    n, a = len(df), len(k)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    msb = (df.groupby("g").y.mean().sub(df.y.mean()) ** 2 * k).sum() / (a - 1)
    msw = df.groupby("g").y.var(ddof=1).mul(k - 1).sum() / (n - a)
    return float((msb - msw) / (msb + (k0 - 1) * msw)) if msb + (k0 - 1) * msw > 0 else 0.0


def potential(y, subj, folds):
    p_null = np.empty(len(y)); p_or = np.empty(len(y))
    for tr, te in folds:
        mu = pd.Series(y[tr]).groupby(subj[tr]).mean()
        p_null[te] = y[tr].mean()
        p_or[te] = pd.Series(subj[te]).map(mu).fillna(y[tr].mean()).to_numpy()
    m_null, m_or = np.abs(p_null - y).mean(), np.abs(p_or - y).mean()
    return float(m_null), float(m_or), float(m_null - m_or)


def identifiability(X, subj, folds):
    hit = 0
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = KNeighborsClassifier(1, algorithm="brute").fit(sc.transform(X[tr]), subj[tr])
        hit += (clf.predict(sc.transform(X[te])) == subj[te]).sum()
    return float(hit / len(subj))


def segment_table(ws, demo, extra_cols=None):
    """Per-record (segment) mean features, demographics, label and subject."""
    F = handcrafted_features(ws.X)
    df = pd.DataFrame(F, columns=[f"f{i}" for i in range(F.shape[1])])
    for j in range(demo.shape[1]):
        df[f"d{j}"] = demo[:, j]
    df["record"], df["subject"], df["y"] = ws.record, ws.subject, ws.y
    for k, v in (extra_cols or {}).items():
        df[k] = v
    return df.groupby("record", as_index=False).agg("first" if False else {**{c: "mean" for c in df.columns if c[0] in "fd" and c[1:].isdigit()},
                                                                         "subject": "first", "y": "first",
                                                                         **{k: "first" for k in (extra_cols or {})}})


def observed(prefix, tgt, models):
    S = {}
    for name in ("vitaldb_summary.json", "ppgbp_summary.json"):
        p = os.path.join(RES, name)
        if os.path.exists(p):
            S.update({(name.split("_")[0], k): v for k, v in json.load(open(p)).items()})
    out = {}
    for m in models:
        leak = S.get((prefix, f"{tgt}_segment_random_{m}_seed0")); hon = S.get((prefix, f"{tgt}_subject_kfold_{m}_seed0"))
        if leak and hon:
            out[m] = {"mae_honest": hon["mae"], "mae_leaky": leak["mae"], "inflation": hon["mae"] - leak["mae"]}
    return out


def record_folds(records, seed=0, k=5):
    """The random-segment folds of run_vitaldb_bp / run_ppgbp, at record level."""
    perm = np.random.default_rng(seed).permutation(len(records))
    f = np.empty(len(records), int); f[perm] = np.arange(len(records)) % k
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def synthetic_sweep(reps=8):
    """Controlled test: vary label clustering (between-subject SD) and identifiability (fingerprint scale)
    independently; for each cohort compute P and a(features) a priori and the observed GBM inflation
    (subject-wise MAE minus record-random MAE, both at record level)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from validate_detectors import make_cohort
    rows = []
    for bsd in (5.0, 10.0, 20.0, 40.0):
        for fp in (0.0, 0.25, 0.5, 1.0):
            for rep in range(reps):
                X, y, subj, rec = make_cohort(7_000_000 + 1000 * int(bsd) + 100 * int(fp * 4) + rep, 0.0,
                                              n_subjects=16, n_records=12, between_sd=bsd, fp_scale=fp)
                F = handcrafted_features(X)
                df = pd.DataFrame(F, columns=[f"f{i}" for i in range(F.shape[1])])
                df["rec"], df["subj"], df["y"] = rec, subj, y
                seg = df.groupby("rec", as_index=False).agg({**{c: "mean" for c in df.columns if c.startswith("f")},
                                                             "subj": "first", "y": "first"})
                Fs, ys, ss = seg[[c for c in seg.columns if c.startswith("f")]].to_numpy(), seg.y.to_numpy(), seg.subj.astype(str).to_numpy()
                leaky = record_folds(seg.rec.to_numpy(), seed=rep)
                honest = list(GroupKFold(5).split(Fs, ys, ss))
                m_null, m_or, P = potential(ys, ss, leaky)
                a = identifiability(Fs, ss, leaky)
                mae, nul = {}, {}
                for name, folds in (("leaky", leaky), ("honest", honest)):
                    pred = np.empty(len(ys)); pn = np.empty(len(ys))
                    for tr, te in folds:
                        pred[te] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                 min_samples_leaf=5, random_state=0).fit(Fs[tr], ys[tr]).predict(Fs[te])
                        pn[te] = ys[tr].mean()
                    mae[name] = float(np.abs(pred - ys).mean()); nul[name] = float(np.abs(pn - ys).mean())
                rows.append({"between_sd": bsd, "fp_scale": fp, "rep": rep, "icc": icc1(ys, ss), "potential": P,
                             "identifiability": a, "predicted_inflation": a * P,
                             "observed_inflation": mae["honest"] - mae["leaky"],
                             "leaky_gain": nul["leaky"] - mae["leaky"], "honest_gain": nul["honest"] - mae["honest"],
                             "mae_model_honest": mae["honest"], "mae_model_leaky": mae["leaky"],
                             "mae_null_honest": nul["honest"], "mae_null_leaky": nul["leaky"], "mae_oracle": m_or,
                             # two-stage prediction of the leaky MAE from the honest MAE (assumption: identified
                             # samples score like the oracle, the rest like the honest model)
                             "predicted_mae_leaky_two_stage": a * m_or + (1 - a) * mae["honest"]})
            sub = [r for r in rows if r["between_sd"] == bsd and r["fp_scale"] == fp]
            print(f"  sweep bsd {bsd:4.0f} fp {fp:.2f}: a {np.mean([r['identifiability'] for r in sub]):.2f}  "
                  f"P {np.mean([r['potential'] for r in sub]):5.2f}  predicted {np.mean([r['predicted_inflation'] for r in sub]):+5.2f}  "
                  f"observed {np.mean([r['observed_inflation'] for r in sub]):+5.2f}  leaky gain {np.mean([r['leaky_gain'] for r in sub]):+5.2f}  "
                  f"honest gain {np.mean([r['honest_gain'] for r in sub]):+5.2f}", flush=True)
    return rows


# Input sets of every model.  "flexible" = can express per-subject offsets (trees, kNN, networks); declared here,
# a priori, and used to split the agreement statistics.  PROSPECTIVE models had not been run on these cohorts when
# their predictions were written (results/leak_risk_preregistered.json, committed before the runs).
INPUTS = {"ridge": ("ppg",), "gbm": ("ppg",), "1d": ("raw",), "demo": ("demo",), "ridge+demo": ("ppg", "demo"),
          "gbm+demo": ("ppg", "demo"), "gbm_demo": ("demo",)}
PROSPECTIVE = {"rf_demo": ("demo",), "knn_demo": ("demo",), "rf+demo": ("ppg", "demo"), "et": ("ppg",),
               "gbm+demo_l50": ("ppg", "demo")}
FLEXIBLE = {"gbm", "1d", "gbm+demo", "gbm_demo", *PROSPECTIVE}


def temporal_potential(y, subj, case, t, folds):
    """MAE of an oracle that predicts each test segment by the label of the nearest-in-time training segment of the
    same case (else the subject oracle, else the training mean): identity *plus* temporal-neighbour leakage."""
    p = np.empty(len(y))
    for tr, te in folds:
        mu_s = pd.Series(y[tr]).groupby(subj[tr]).mean()
        g = {c: (t[tr][case[tr] == c], y[tr][case[tr] == c]) for c in np.unique(case[tr])}
        for i in te:
            if case[i] in g:
                tt, yy = g[case[i]]
                p[i] = yy[np.argmin(np.abs(tt - t[i]))]
            else:
                p[i] = mu_s.get(subj[i], y[tr].mean())
    return float(np.abs(p - y).mean())


def cohort_stats():
    from run_vitaldb_bp import load_vitaldb
    from run_ppgbp import load_ppgbp
    from sklearn.model_selection import KFold
    out = {}
    for prefix, loader in (("vitaldb", load_vitaldb), ("ppgbp", load_ppgbp)):
        for tgt in ("sbp", "dbp"):
            res = loader(tgt)
            ws, demo = res[0], res[1]
            if prefix == "ppgbp":
                # PPG-BP stores one record per subject; its unit of splitting is the segment (one row each)
                F = handcrafted_features(ws.X)
                seg = pd.DataFrame(F, columns=[f"f{i}" for i in range(F.shape[1])])
                for j in range(demo.shape[1]):
                    seg[f"d{j}"] = demo[:, j]
                seg["subject"], seg["y"], seg["record"] = ws.subject, ws.y, np.arange(len(ws)).astype(str)
                raw = ws.X
            else:
                case, tt = res[2], res[3]
                seg = segment_table(ws, demo, {"case": case, "t": tt})
                first = pd.Series(np.arange(len(ws))).groupby(ws.record).first()      # first 5-s window of each segment
                raw = ws.X[first.loc[seg.record].to_numpy()]
            y, subj = seg.y.to_numpy(), seg.subject.astype(str).to_numpy()
            folds = record_folds(seg.record.to_numpy()) if prefix == "vitaldb" else \
                [(tr, te) for tr, te in KFold(5, shuffle=True, random_state=0).split(seg)]
            m_null, m_or, P = potential(y, subj, folds)
            Fp = seg[[c for c in seg.columns if c.startswith("f")]].to_numpy()
            Fd = seg[[c for c in seg.columns if c.startswith("d")]].to_numpy()
            ident = {"ppg": identifiability(Fp, subj, folds), "demo": identifiability(Fd, subj, folds),
                     "ppg+demo": identifiability(np.hstack([Fp, Fd]), subj, folds),
                     "raw": identifiability(raw, subj, folds)}
            d = {"icc": icc1(y, subj), "mae_null": m_null, "mae_subject_oracle": m_or, "potential": P,
                 "identifiability": ident, "n_records": int(len(seg)), "n_subjects": int(len(set(subj))),
                 "chance_identifiability": float(1.0 / len(set(subj)))}
            if prefix == "vitaldb":
                m_t = temporal_potential(y, subj, seg.case.astype(str).to_numpy(), seg.t.to_numpy(), folds)
                d.update(mae_temporal_oracle=m_t, potential_temporal=m_null - m_t)
            out[f"{prefix}_{tgt}"] = d
            print(f"{prefix} {tgt}: ICC {d['icc']:.2f} P {P:.2f}" + (f" P_time {d['potential_temporal']:.2f}" if "potential_temporal" in d else "")
                  + "  a: " + " ".join(f"{k} {v:.2f}" for k, v in ident.items()), flush=True)
    return out


def predict(stats, inputs):
    """Primary prediction a(joint inputs) * P; secondary a(max over input blocks) * P."""
    key = "+".join(inputs)
    a_joint = stats["identifiability"][key] if key in stats["identifiability"] else None
    a_max = max(stats["identifiability"][b] for b in inputs)
    return {"inputs": key, "identifiability": a_joint, "identifiability_max_block": a_max,
            "predicted_inflation": a_joint * stats["potential"], "predicted_inflation_max_block": a_max * stats["potential"]}


def main() -> None:
    import argparse
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["cohorts", "sweep"])
    ap.add_argument("--preregister", action="store_true", help="write predictions for PROSPECTIVE models and stop")
    args = ap.parse_args()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "cohorts" in args.parts or args.preregister:
        out["cohorts"] = cohort_stats()
    if args.preregister:
        reg = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "rule": "predicted inflation = a(joint inputs) * P on the random-segment folds; a = 1-NN subject "
                       "identification accuracy; P = MAE(train-mean null) - MAE(subject oracle). Observed inflation will be "
                       "subject-wise MAE minus random-segment MAE from run_ppgbp.py / run_vitaldb_bp.py.",
               "predictions": {}}
        for key, st in out["cohorts"].items():
            for m, inp in PROSPECTIVE.items():
                reg["predictions"][f"{key}/{m}"] = predict(st, inp)
        json.dump(reg, open(os.path.join(RES, "leak_risk_preregistered.json"), "w"), indent=2)
        json.dump(out, open(OUT, "w"), indent=2)
        print("preregistered", len(reg["predictions"]), "predictions")
        return
    # assemble predicted vs observed for every model that has results (retrospective + prospective)
    pre = json.load(open(os.path.join(RES, "leak_risk_preregistered.json"))) if os.path.exists(
        os.path.join(RES, "leak_risk_preregistered.json")) else {"predictions": {}}
    for key, st in out.get("cohorts", {}).items():
        prefix, tgt = key.split("_")
        rows = {}
        for m, o in observed(prefix, tgt, list(INPUTS) + list(PROSPECTIVE)).items():
            pr = pre["predictions"].get(f"{key}/{m}") if m in PROSPECTIVE else predict(st, INPUTS[m])
            if pr is None:
                continue
            rows[m] = {**o, **pr, "flexible": m in FLEXIBLE, "prospective": m in PROSPECTIVE}
        st["models"] = rows
        for m, r in rows.items():
            print(f"  {key:12s} {m:13s} {'P' if r['prospective'] else ' '} observed {r['inflation']:+6.2f}  "
                  f"predicted {r['predicted_inflation']:+6.2f}  (max-block {r['predicted_inflation_max_block']:+6.2f})")
    if "sweep" in args.parts:
        out["synthetic_sweep"] = synthetic_sweep()
    json.dump(out, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
