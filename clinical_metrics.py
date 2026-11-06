"""
clinical_metrics.py
===================
Clinical anchoring of the benchmark numbers, computed from the out-of-fold
prediction files in results/:

  * hypo- (< 70 mg/dL) and hyper-glycaemia (> 180 mg/dL) event sensitivity /
    specificity at reading level, for every model and for the null;
  * time-in-range (70-180) agreement;
  * reference-noise floors: the MAE / MARD a PERFECT estimator would show
    against a reference that is itself noisy -- ISO 15197:2013 meter limits
    (95 % within +/-15 mg/dL below 100 mg/dL, +/-15 % at or above) for the
    Mendeley fingerstick labels, and a Dexcom G6-like CGM (MARD 9.0 %) for
    BIG IDEAs -- simulated by adding that noise to the reference and scoring
    the clean values against it;
  * Parkes/Consensus error-grid zones (type 1 grid) when ppgbench.metrics is
    importable (falls back to Clarke only otherwise).

    python clinical_metrics.py        # -> results/clinical_metrics.json + printed table
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from plot_clarke_error_grid import summarize_clarke_zones

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
HYPO, HYPER = 70.0, 180.0


def _load(pattern):
    files = sorted(glob.glob(os.path.join(RES, pattern)))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    return df.groupby(["record", "subject"], as_index=False).agg(y=("y", "first"), pred=("pred", "mean"))


def event_metrics(y, p):
    out = {}
    for name, mask_y, mask_p in (("hypo", y < HYPO, p < HYPO), ("hyper", y > HYPER, p > HYPER)):
        tp = int((mask_y & mask_p).sum()); fn = int((mask_y & ~mask_p).sum())
        fp = int((~mask_y & mask_p).sum()); tn = int((~mask_y & ~mask_p).sum())
        out[f"{name}_n_events"] = int(mask_y.sum())
        out[f"{name}_sensitivity"] = tp / (tp + fn) if tp + fn else None
        out[f"{name}_specificity"] = tn / (tn + fp) if tn + fp else None
        out[f"{name}_detected"] = tp
    in_y, in_p = (y >= HYPO) & (y <= HYPER), (p >= HYPO) & (p <= HYPER)
    out["tir_agreement"] = float((in_y == in_p).mean())
    return out


def noise_floor(y, kind, rng, n_rep=200):
    """MAE/MARD of the clean reference against a reference corrupted with the
    device's own error model (what a perfect estimator would score)."""
    maes, mards = [], []
    for _ in range(n_rep):
        if kind == "iso15197":                       # 95 % within +/-15 mg/dL (<100) or +/-15 % (>=100): sd = limit/1.96
            sd = np.where(y < 100, 15.0, 0.15 * y) / 1.96
        elif kind == "g6":                           # MARD 9 % -> Gaussian sd = MARD * sqrt(pi/2) * y
            sd = 0.09 * np.sqrt(np.pi / 2) * y
        else:
            raise ValueError(kind)
        e = rng.normal(0, sd)
        maes.append(np.abs(e).mean()); mards.append((np.abs(e) / y).mean() * 100)
    return {"kind": kind, "mae": float(np.mean(maes)), "mard_pct": float(np.mean(mards))}


def parkes_or_none(y, p):
    try:
        from ppgbench.metrics import parkes_zone  # noqa: F401
        zones = np.array([parkes_zone(a, b) for a, b in zip(y, p)])
        return {f"parkes_{z}_pct": float((zones == z).mean() * 100) for z in "ABCDE"}
    except Exception:
        return {}


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for exp, patterns in (
        ("mendeley", {m: f"subject_cv_{m}_seed*.csv" for m in ["mean", "ridge", "demo", "1d", "2d", "3d", "fusion", "cnnlstm", "resnet1d", "gbm", "papagei", "anyppg", "pulseppg"]}),
        ("bigideas", {m: f"bigideas_within_{m}_seed*.csv" for m in ["mean", "ridge", "1d", "2d", "3d", "fusion", "tod", "persist",
                                                                    "ridge_res", "1d_res", "2d_res", "3d_res", "fusion_res", "papagei", "anyppg", "pulseppg", "papagei_res", "anyppg_res", "pulseppg_res"]}),
        ("stuba", {m: f"subject_cv_stuba_IR1_{m}_seed*.csv" for m in ["mean", "ridge", "gbm", "1d", "2d", "3d", "fusion", "papagei", "anyppg", "pulseppg"]}),
    ):
        for model, pat in patterns.items():
            df = _load(pat)
            if df is None:
                continue
            y, p = df.y.to_numpy(), df.pred.to_numpy()
            d = event_metrics(y, p)
            d.update({f"clarke_{z}": v for z, v in summarize_clarke_zones(y.astype(np.float32), p.astype(np.float32)).items()
                      if z.endswith("_pct")})
            d.update(parkes_or_none(y, p))
            out[f"{exp}/{model}"] = d
        df = _load(patterns["mean"])
        if df is not None:
            y = df.y.to_numpy()
            out[f"{exp}/noise_floor"] = noise_floor(y, "g6" if exp == "bigideas" else "iso15197", rng)
            out[f"{exp}/reference_range"] = {"n": int(len(y)), "hypo_n": int((y < HYPO).sum()), "hyper_n": int((y > HYPER).sum()),
                                             "in_range_pct": float(((y >= HYPO) & (y <= HYPER)).mean() * 100)}
    json.dump(out, open(os.path.join(RES, "clinical_metrics.json"), "w"), indent=2)
    for k, v in out.items():
        if "sensitivity" in str(v):
            print(f"{k:22s} hypo {v['hypo_detected']}/{v['hypo_n_events']}  hyper {v['hyper_detected']}/{v['hyper_n_events']}  "
                  f"TIR agree {v['tir_agreement']:.3f}  Clarke A {v['clarke_zone_A_pct']:.1f}" +
                  (f"  Parkes A {v['parkes_A_pct']:.1f}" if "parkes_A_pct" in v else ""))
        else:
            print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
