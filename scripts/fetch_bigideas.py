#!/usr/bin/env python
"""
Stream the PhysioNet BIG IDEAs Glycemic Variability & Wearable Device dataset
(v1.1.3) into compact, CGM-aligned window files.

Why this exists
---------------
Each subject's raw Empatica E4 BVP file is ~1.4 GB of CSV (34 GB uncompressed for
16 subjects).  This machine has ~11 GB of free disk, so we never unpack the
archive.  Instead we read each member directly from the zip (or from an
already-downloaded folder), keep only the samples that fall within
+/- WIN_PRE/WIN_POST seconds of each Dexcom G6 estimated glucose value (EGV),
and store them as NumPy arrays.

Outputs (one file per subject, ~100 MB)
---------------------------------------
    <out_dir>/sub_XXX.npz
        egv_time      (n,)          float64  POSIX seconds of the CGM reading
        glucose       (n,)          float32  mg/dL
        roc           (n,)          float32  Dexcom rate of change (mg/dL/min, NaN if absent)
        bvp           (n, 7680)     float32  64 Hz BVP, NaN where the device was off
        bvp_coverage  (n,)          float32  fraction of the window with BVP data
        acc           (n, 3, 3840)  int16    32 Hz accelerometer (1/64 g), -32768 = missing
        acc_coverage  (n,)          float32
        eda           (n, 480)      float32  4 Hz
        temp          (n, 480)      float32  4 Hz
        hr            (n, 120)      float32  1 Hz (Empatica-derived)
    <out_dir>/sub_XXX_dexcom.csv  copy of the EGV rows
    <out_dir>/sub_XXX_food.csv    copy of the food log
    <out_dir>/sub_XXX_summary.json

Usage
-----
    python scripts/fetch_bigideas.py --zip data.nosync/bigideas/bigideas-1.1.3.zip \
        --out data.nosync/bigideas --subjects 001 002 ... 016
    python scripts/fetch_bigideas.py --dir physionet.org/files/big-ideas-glycemic-wearable/1.1.3 \
        --out data.nosync/bigideas --subjects 001
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fetch_bigideas")

WIN_PRE = 60.0    # seconds before the CGM timestamp
WIN_POST = 60.0   # seconds after the CGM timestamp
FS = {"BVP": 64.0, "ACC": 32.0, "EDA": 4.0, "TEMP": 4.0, "HR": 1.0}


# --------------------------------------------------------------------------- #
# Source abstraction: zip archive or plain directory
# --------------------------------------------------------------------------- #
class Source:
    def __init__(self, zip_path: Path | None, dir_path: Path | None):
        self.zf = zipfile.ZipFile(zip_path) if zip_path else None
        self.dir = dir_path
        if self.zf is not None:
            self.names = self.zf.namelist()

    def open(self, subject: str, kind: str):
        """Return a binary file object for e.g. 001/BVP_001.csv."""
        rel = f"{subject}/{kind}_{subject}.csv"
        if self.zf is not None:
            cands = [n for n in self.names if n.endswith(rel)]
            if not cands:
                raise FileNotFoundError(rel)
            return self.zf.open(cands[0], "r")
        p = self.dir / rel
        if not p.exists():
            raise FileNotFoundError(p)
        return open(p, "rb")

    def exists(self, subject: str, kind: str) -> bool:
        rel = f"{subject}/{kind}_{subject}.csv"
        if self.zf is not None:
            return any(n.endswith(rel) for n in self.names)
        return (self.dir / rel).exists()


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _to_epoch(series: pd.Series, fmt: str | None) -> np.ndarray:
    ts = pd.to_datetime(series.str.strip(), format=fmt, errors="coerce")
    # pandas >= 2 may parse to a non-nanosecond resolution, so never assume
    # astype("int64") yields nanoseconds: divide by an explicit Timedelta.
    return ((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(np.float64)


def read_dexcom(src: Source, subject: str) -> pd.DataFrame:
    with src.open(subject, "Dexcom") as f:
        df = pd.read_csv(f, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    egv = df[df["Event Type"].str.strip() == "EGV"].copy()
    tcol = [c for c in egv.columns if c.startswith("Timestamp")][0]
    egv["t"] = _to_epoch(egv[tcol], "%Y-%m-%d %H:%M:%S")
    g = pd.to_numeric(egv["Glucose Value (mg/dL)"].str.strip(), errors="coerce")
    # Dexcom encodes out-of-range values as 'Low' (<40) / 'High' (>400)
    raw = egv["Glucose Value (mg/dL)"].str.strip().str.lower()
    g = g.where(~raw.eq("low"), 39.0)
    g = g.where(~raw.eq("high"), 401.0)
    egv["glucose"] = g.astype(np.float32)
    roc_col = "Glucose Rate of Change (mg/dL/min)"
    egv["roc"] = pd.to_numeric(egv[roc_col], errors="coerce").astype(np.float32) if roc_col in egv else np.nan
    egv = egv.dropna(subset=["t", "glucose"]).sort_values("t").reset_index(drop=True)
    return egv[["t", "glucose", "roc"]]


def read_signal(src: Source, subject: str, kind: str, cols: list[str], fmt: str,
                chunksize: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Stream a (possibly 1.4 GB) CSV and return (t_epoch float64, values float32[,k])."""
    t_parts, v_parts = [], []
    t0 = time.time()
    nrows = 0
    with src.open(subject, kind) as f:
        reader = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", newline=""),
                             chunksize=chunksize, dtype=str, skipinitialspace=True)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            t_parts.append(_to_epoch(chunk["datetime"], fmt))
            vals = np.column_stack([pd.to_numeric(chunk[c], errors="coerce").to_numpy(np.float32)
                                    for c in cols])
            v_parts.append(vals.squeeze() if len(cols) == 1 else vals)
            nrows += len(chunk)
            log.info("  %s_%s: %d rows (%.0f s)", kind, subject, nrows, time.time() - t0)
    t = np.concatenate(t_parts)
    v = np.concatenate(v_parts)
    ok = np.isfinite(t)
    if not ok.all():
        log.warning("  %s_%s: %d unparseable timestamps dropped", kind, subject, (~ok).sum())
        t, v = t[ok], v[ok]
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def window_stack(t: np.ndarray, v: np.ndarray, centers: np.ndarray, fs: float,
                 fill, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Place samples into fixed-length windows [c-WIN_PRE, c+WIN_POST) at nominal fs."""
    n_len = int(round((WIN_PRE + WIN_POST) * fs))
    k = 1 if v.ndim == 1 else v.shape[1]
    out = np.full((len(centers), k, n_len), fill, dtype=dtype)
    cov = np.zeros(len(centers), dtype=np.float32)
    for i, c in enumerate(centers):
        a = np.searchsorted(t, c - WIN_PRE, side="left")
        b = np.searchsorted(t, c + WIN_POST, side="left")
        if b <= a:
            continue
        idx = np.rint((t[a:b] - (c - WIN_PRE)) * fs).astype(np.int64)
        keep = (idx >= 0) & (idx < n_len)
        idx = idx[keep]
        vals = v[a:b][keep]
        if k == 1:
            out[i, 0, idx] = vals
        else:
            out[i, :, idx] = vals          # advanced indexing yields (len(idx), k); vals is (len(idx), k)
        cov[i] = len(np.unique(idx)) / n_len
    return (out[:, 0, :] if k == 1 else out), cov


# --------------------------------------------------------------------------- #
# Per-subject driver
# --------------------------------------------------------------------------- #
def process_subject(src: Source, subject: str, out_dir: Path, force: bool = False,
                    signals: tuple[str, ...] = ("BVP", "ACC", "EDA", "TEMP", "HR")) -> dict:
    out_npz = out_dir / f"sub_{subject}.npz"
    if out_npz.exists() and not force:
        log.info("sub_%s already done; skipping", subject)
        return json.loads((out_dir / f"sub_{subject}_summary.json").read_text())

    t_start = time.time()
    egv = read_dexcom(src, subject)
    centers = egv["t"].to_numpy()
    log.info("sub_%s: %d EGV readings, glucose %.0f-%.0f mg/dL", subject, len(egv),
             egv.glucose.min(), egv.glucose.max())
    egv.to_csv(out_dir / f"sub_{subject}_dexcom.csv", index=False)
    if src.exists(subject, "Food_Log"):
        with src.open(subject, "Food_Log") as f, open(out_dir / f"sub_{subject}_food.csv", "wb") as g:
            shutil.copyfileobj(f, g)

    arrays: dict[str, np.ndarray] = {
        "egv_time": centers, "glucose": egv.glucose.to_numpy(np.float32),
        "roc": egv.roc.to_numpy(np.float32),
    }
    summary = {"subject": subject, "n_egv": int(len(egv)),
               "glucose_min": float(egv.glucose.min()), "glucose_max": float(egv.glucose.max()),
               "glucose_mean": float(egv.glucose.mean()), "glucose_std": float(egv.glucose.std()),
               "egv_start": float(centers.min()), "egv_end": float(centers.max())}

    spec = {
        "BVP": (["bvp"], "%Y-%m-%d %H:%M:%S.%f", np.nan, np.float32),
        "ACC": (["acc_x", "acc_y", "acc_z"], "%Y-%m-%d %H:%M:%S.%f", -32768, np.int16),
        "EDA": (["eda"], "%Y-%m-%d %H:%M:%S.%f", np.nan, np.float32),
        "TEMP": (["temp"], "%Y-%m-%d %H:%M:%S.%f", np.nan, np.float32),
        "HR": (["hr"], "%Y-%m-%d %H:%M:%S", np.nan, np.float32),
    }
    for kind, (cols, fmt, fill, dtype) in spec.items():
        if kind not in signals:
            continue
        if not src.exists(subject, kind):
            log.warning("sub_%s: %s missing", subject, kind)
            continue
        t, v = read_signal(src, subject, kind, cols, fmt)
        if kind == "ACC":
            v = np.nan_to_num(v, nan=-32768).astype(np.int16)
        win, cov = window_stack(t, v, centers, FS[kind], fill, dtype)
        key = kind.lower()
        arrays[key] = win
        arrays[f"{key}_coverage"] = cov
        summary[f"{key}_rows"] = int(len(t))
        summary[f"{key}_start"] = float(t[0]); summary[f"{key}_end"] = float(t[-1])
        summary[f"{key}_full_windows"] = int((cov >= 0.99).sum())
        summary[f"{key}_mean_coverage"] = float(cov.mean())
        del t, v
    np.savez(out_npz, **arrays)
    summary["seconds"] = round(time.time() - t_start, 1)
    summary["npz_mb"] = round(out_npz.stat().st_size / 1e6, 1)
    (out_dir / f"sub_{subject}_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("sub_%s done in %.0f s -> %s (%.0f MB)", subject, summary["seconds"], out_npz,
             summary["npz_mb"])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="path to the PhysioNet zip archive")
    ap.add_argument("--dir", type=Path, help="path to an unpacked 1.1.3/ folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subjects", nargs="+", default=[f"{i:03d}" for i in range(1, 17)])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--signals", nargs="+", default=["BVP", "ACC", "EDA", "TEMP", "HR"],
                    help="which E4 streams to extract (BVP alone is enough for the paper)")
    args = ap.parse_args()
    if not args.zip and not args.dir:
        sys.exit("need --zip or --dir")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    src = Source(args.zip, args.dir)
    summaries = []
    for s in args.subjects:
        try:
            summaries.append(process_subject(src, s, args.out, args.force, tuple(args.signals)))
        except FileNotFoundError as e:
            log.error("sub_%s: missing %s", s, e)
    (args.out / "summary_all.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
