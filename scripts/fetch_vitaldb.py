"""
fetch_vitaldb.py
================
Extract quality-controlled 10-s finger-PPG segments with beat-wise arterial
blood-pressure labels from the open VitalDB intra-operative database
(Lee et al., Sci. Data 9:279, 2022; https://vitaldb.net).

For a seeded random sample of cases that carry both ``SNUADC/PLETH`` and
``SNUADC/ART`` (500 Hz, read at 100 Hz through the official ``vitaldb``
client), candidate segments are placed every ``--spacing-s`` seconds between
10 min after the start and 10 min before the end of the recording.  A segment
is kept only if

* neither channel contains NaN, and the arterial trace never drops below
  20 mmHg (flush / zeroing artefacts);
* the arterial trace has 6-30 beats (36-180 bpm) with a stable systolic
  value (inter-quartile range of beat peaks < 15 mmHg);
* SBP 70-200 mmHg, DBP 30-120 mmHg, pulse pressure >= 20 mmHg;
* the PPG is not flat or clipped and its beat count is within 3 of the
  arterial beat count (same heart, both channels valid).

Label = median beat systolic peak (SBP) and median inter-beat trough (DBP)
over the 10-s segment.  At most ``--max-per-case`` segments are kept per case
(random, seeded).  Output: one ``<caseid>.npz`` per case with ``ppg``
(n x 1000 at 100 Hz), ``sbp``, ``dbp``, ``t`` (s from case start), plus
``index.csv`` with caseid, subjectid, age, sex, height, weight, n_segments.

    python scripts/fetch_vitaldb.py --n-cases 600 --workers 4
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

log = logging.getLogger("fetch_vitaldb")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data.nosync", "vitaldb")
FS = 100
SEG = 10 * FS


def label_segment(ppg: np.ndarray, art: np.ndarray):
    """Return (sbp, dbp) or None if the segment fails quality control."""
    if np.isnan(ppg).any() or np.isnan(art).any() or art.min() < 20:
        return None
    pk, _ = find_peaks(art, distance=int(0.33 * FS), prominence=15)
    if not 6 <= len(pk) <= 30:
        return None
    peaks = art[pk]
    if np.subtract(*np.percentile(peaks, [75, 25])) >= 15:
        return None
    troughs = np.array([art[a:b].min() for a, b in zip(pk[:-1], pk[1:])])
    sbp, dbp = float(np.median(peaks)), float(np.median(troughs))
    if not (70 <= sbp <= 200 and 30 <= dbp <= 120 and sbp - dbp >= 20):
        return None
    if ppg.std() < 1e-3 or np.mean(ppg >= ppg.max() - 1e-6) > 0.05 or np.mean(ppg <= ppg.min() + 1e-6) > 0.05:
        return None
    x = ppg - np.convolve(ppg, np.ones(FS) / FS, mode="same")
    ppk, _ = find_peaks(x, distance=int(0.33 * FS), prominence=0.3 * x.std())
    if abs(len(ppk) - len(pk)) > 3:
        return None
    return sbp, dbp


def process_case(caseid: int, spacing_s: float, max_per_case: int, seed: int):
    import vitaldb
    path = os.path.join(OUT, f"{caseid}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return caseid, int(len(d["sbp"]))
    x = vitaldb.load_case(int(caseid), ["SNUADC/PLETH", "SNUADC/ART"], 1 / FS)
    if x is None or len(x) < 30 * 60 * FS:
        np.savez_compressed(path, ppg=np.zeros((0, SEG), np.float32), sbp=np.zeros(0), dbp=np.zeros(0), t=np.zeros(0))
        return caseid, 0
    ppg_all, art_all = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    starts = np.arange(10 * 60 * FS, len(x) - 10 * 60 * FS - SEG, int(spacing_s * FS))
    keep = []
    for s in starts:
        lab = label_segment(ppg_all[s:s + SEG], art_all[s:s + SEG])
        if lab is not None:
            keep.append((s, *lab))
    rng = np.random.default_rng(seed + int(caseid))
    if len(keep) > max_per_case:
        keep = [keep[i] for i in sorted(rng.choice(len(keep), max_per_case, replace=False))]
    ppg = np.array([ppg_all[s:s + SEG] for s, _, _ in keep], np.float32).reshape(-1, SEG)
    np.savez_compressed(path, ppg=ppg, sbp=np.array([k[1] for k in keep]), dbp=np.array([k[2] for k in keep]),
                        t=np.array([k[0] / FS for k in keep]))
    return caseid, len(keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cases", type=int, default=600)
    ap.add_argument("--spacing-s", type=float, default=120.0)
    ap.add_argument("--max-per-case", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    os.makedirs(OUT, exist_ok=True)
    import vitaldb
    ids = sorted(int(i) for i in vitaldb.find_cases(["SNUADC/PLETH", "SNUADC/ART"]))
    clin_path = os.path.join(OUT, "clinical.csv")
    clin = pd.read_csv(clin_path) if os.path.exists(clin_path) else pd.read_csv("https://api.vitaldb.net/cases")
    clin.to_csv(clin_path, index=False)
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(ids, min(args.n_cases, len(ids)), replace=False).tolist())
    log.info("%d eligible cases; sampling %d", len(ids), len(pick))
    counts = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_case, c, args.spacing_s, args.max_per_case, args.seed): c for c in pick}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                c, n = f.result()
                counts[c] = n
            except Exception as e:                       # one bad case must not stop the run
                log.warning("case %s failed: %s", futs[f], e)
                counts[futs[f]] = 0
            if i % 25 == 0:
                log.info("%d/%d cases, %d segments so far", i, len(pick), sum(counts.values()))
    idx = clin[clin.caseid.isin(list(counts))][["caseid", "subjectid", "age", "sex", "height", "weight"]].copy()
    idx["n_segments"] = idx.caseid.map(counts)
    idx.to_csv(os.path.join(OUT, "index.csv"), index=False)
    ok = idx[idx.n_segments > 0]
    log.info("done: %d cases with segments, %d patients, %d segments", len(ok), ok.subjectid.nunique(), int(ok.n_segments.sum()))


if __name__ == "__main__":
    main()
