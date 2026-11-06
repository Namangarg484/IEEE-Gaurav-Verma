"""
run_vitaldb_bp.py
=================
The leakage audit on a large multi-reading blood-pressure cohort: finger PPG
-> arterial-line SBP/DBP in VitalDB (Lee et al., Sci. Data 9:279, 2022),
extracted by ``scripts/fetch_vitaldb.py`` (10-s segments, beat-wise labels,
quality-controlled; many segments per surgical case, several cases for some
patients).

Each segment is band-passed 0.5-8 Hz, resampled to 64 Hz and cut into three
5-s windows (50 % overlap); window predictions are averaged per segment, the
unit of evaluation.  Groups: record = segment, case = surgical case,
subject = VitalDB ``subjectid`` (patient).

Protocols (the ladder)
    segment_random  5-fold over segments: a patient's other segments, minutes
                    apart in the same surgery, are in training (identity +
                    temporal leak) -- the split behind most published
                    cuffless-BP accuracies on VitalDB / MIMIC-derived data
    case_kfold      5-fold grouped by case (leaks only patients with >1 case)
    subject_kfold   5-fold grouped by patient (honest, calibration-free)
    personal        chronological within every case: first half of each
                    case's segments train, second half test (calibration-
                    based use); null = each case's own training mean
    personal_res    as personal, but models learn the deviation from each
                    case's own training mean and add it back (they receive
                    the same calibration information as the null)

Models: train-mean null (subject_kfold etc.) or case-mean null (personal);
ridge on nine hand-crafted PPG features; ridge on age/sex/height/weight;
both; gradient boosting on both; 1-D CNN.  Metrics per segment with the
subject-cluster bootstrap; mean error (ME) and SD of error are reported
against the AAMI/ISO 81060-2 limits (|ME| <= 5, SD <= 8 mmHg).

    python run_vitaldb_bp.py --targets sbp dbp
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ppg_data import WindowSet, _zscore_rows, handcrafted_features, make_views, to_common_rate
from ppgbench.core.signal import sliding_windows
from run_ppgbp import fit_classical
from run_subject_cv import OUT_DIR, device, merge_json, train_fold
from validate_detectors import cluster_ci

log = logging.getLogger("vitaldb_bp")
ROOT = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join(ROOT, "data.nosync", "vitaldb")


def load_vitaldb(target: str):
    idx = pd.read_csv(os.path.join(VDIR, "index.csv"))
    idx = idx[idx.n_segments > 0]
    Xs, ys, rec, case, subj, demo, tt = [], [], [], [], [], [], []
    for _, r in idx.iterrows():
        d = np.load(os.path.join(VDIR, f"{int(r.caseid)}.npz"))
        dem = [float(r.age), float(r.height), float(r.weight), 1.0 if str(r.sex).upper().startswith("M") else 0.0]
        for i in range(len(d["sbp"])):
            W = sliding_windows(to_common_rate(d["ppg"][i].astype(np.float64), 100), 320, 160)
            key = f"{int(r.caseid)}_{i:03d}"
            Xs.append(W); ys += [float(d[target][i])] * len(W); rec += [key] * len(W)
            case += [str(int(r.caseid))] * len(W); subj += [str(int(r.subjectid))] * len(W)
            demo += [dem] * len(W); tt += [float(d["t"][i])] * len(W)
    X = _zscore_rows(np.concatenate(Xs).astype(np.float32))
    meta = pd.DataFrame({"record": rec, "case": case, "subject": subj, "t": tt}).drop_duplicates("record")
    ws = WindowSet(X, np.array(ys, np.float32), np.array(subj), np.array(rec), meta, source=f"VitalDB ({target.upper()})")
    D = np.array(demo, np.float32)
    D[~np.isfinite(D)] = np.nanmean(D, axis=0)[np.where(~np.isfinite(D))[1]]
    return ws, D, np.array(case), np.array(tt)


def folds_for(ws, case, t, protocol, seed):
    rec_codes, rec_inv = np.unique(ws.record, return_inverse=True)
    if protocol == "segment_random":
        perm = np.random.default_rng(seed).permutation(len(rec_codes))
        fold_of_rec = np.empty(len(rec_codes), int); fold_of_rec[perm] = np.arange(len(rec_codes)) % 5
        f = fold_of_rec[rec_inv]
        return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]
    if protocol == "case_kfold":
        return list(GroupKFold(5).split(ws.X, ws.y, case))
    if protocol == "subject_kfold":
        return list(GroupKFold(5).split(ws.X, ws.y, ws.subject))
    if protocol in ("personal", "personal_res"):
        rk = pd.DataFrame({"case": case, "t": t}).groupby("case").t.rank(method="dense", pct=True).to_numpy()
        return [(np.where(rk <= 0.5)[0], np.where(rk > 0.5)[0])]
    raise ValueError(protocol)


def segment_level(pred, ws, case):
    df = pd.DataFrame({"record": ws.record, "subject": ws.subject, "case": case, "y": ws.y, "pred": pred})
    return df.dropna().groupby(["record", "subject", "case"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))


def bp_summary(df, null_df, rng):
    e = df.pred.to_numpy() - df.y.to_numpy()
    y = df.y.to_numpy()
    out = {"n_segments": int(len(df)), "n_subjects": int(df.subject.nunique()), "n_cases": int(df.case.nunique()),
           "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "me": float(e.mean()),
           "sde": float(e.std()), "pearson_r": float(np.corrcoef(y, df.pred)[0, 1]) if df.pred.std() > 0 else None,
           "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    w = [np.corrcoef(g.y, g.pred)[0, 1] for _, g in df.groupby("case") if len(g) > 5 and g.y.std() > 0 and g.pred.std() > 0]
    out["within_case_r_median"] = float(np.median(w)) if w else None
    if null_df is not None:
        m = df.merge(null_df[["record", "pred"]], on="record", suffixes=("", "_null"))
        d, ci = cluster_ci(m.y.to_numpy(), m.pred.to_numpy(), m.pred_null.to_numpy(), m.subject.to_numpy(), rng)
        out.update(diff_vs_null=d, diff_ci=[float(ci[0]), float(ci[1])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["sbp", "dbp"])
    ap.add_argument("--protocols", nargs="+", default=["segment_random", "case_kfold", "subject_kfold", "personal"])
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "demo", "ridge+demo", "gbm+demo", "1d"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    path = os.path.join(OUT_DIR, "vitaldb_summary.json")
    dev = device()
    for target in args.targets:
        ws, demo, case, t = load_vitaldb(target)
        log.info("VitalDB %s: %d windows, %d segments, %d cases, %d patients", target.upper(), len(ws),
                 len(np.unique(ws.record)), len(np.unique(case)), len(np.unique(ws.subject)))
        feats = handcrafted_features(ws.X)
        F = {"ridge": feats, "demo": demo, "ridge+demo": np.hstack([feats, demo]), "gbm+demo": np.hstack([feats, demo]), "nn1": ws.X,
             "gbm": feats}
        views = make_views(ws.X, which=("x1",)) if "1d" in args.models else None
        for protocol in args.protocols:
            null_df = None
            for name in args.models:
                for seed in (args.seeds if name == "1d" else [0]):
                    folds = folds_for(ws, case, t, protocol, seed)
                    pred = np.full(len(ws), np.nan, np.float32)
                    t0 = time.time()
                    for tr, te in folds:
                        if protocol.startswith("personal"):
                            mu = pd.Series(ws.y[tr]).groupby(case[tr]).mean()
                            base = pd.Series(case).map(mu).fillna(ws.y[tr].mean()).to_numpy().astype(np.float32)
                        if name == "mean" and protocol.startswith("personal"):   # each case's own training mean
                            pred[te] = base[te]
                            continue
                        # personal_res: learn the deviation from each case's own training mean (calibration input)
                        y_fit = ws.y - base if protocol == "personal_res" else ws.y
                        if name == "1d":
                            p_ = train_fold("1d", views, y_fit, tr, te, args.epochs, 128, 1e-3, seed, dev)
                        else:
                            p_ = fit_classical(name, F.get(name), y_fit, tr, te)
                        pred[te] = p_ + (base[te] if protocol == "personal_res" else 0.0)
                    df = segment_level(pred, ws, case)
                    tag = f"{target}_{protocol}_{name}_seed{seed}"
                    df.to_csv(os.path.join(OUT_DIR, f"vitaldb_{tag}.csv"), index=False)
                    m = bp_summary(df, None if name == "mean" else null_df, np.random.default_rng(seed))
                    if name == "mean":
                        null_df = df
                    m.update(target=target, protocol=protocol, model=name, seed=seed, seconds=round(time.time() - t0, 1))
                    merge_json(path, tag, m)
                    log.info("RESULT %-38s MAE %.2f  ME %+.2f SD %.2f  r %.3f  wr %s  delta %s", tag, m["mae"], m["me"], m["sde"],
                             m["pearson_r"] or np.nan, f"{m['within_case_r_median']:.2f}" if m["within_case_r_median"] is not None else "--",
                             f"{m['diff_vs_null']:+.2f} [{m['diff_ci'][0]:+.2f},{m['diff_ci'][1]:+.2f}]" if "diff_ci" in m else "--")


if __name__ == "__main__":
    main()
