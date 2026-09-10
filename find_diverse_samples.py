"""
find_diverse_samples.py
=======================
Scan the dataset to locate window indices that cover the widest range of
unique glucose levels (one representative window per target glucose bucket).

Prints the selected indices to stdout and saves them to diverse_samples.json.

Usage:
    python find_diverse_samples.py
    python find_diverse_samples.py --csv /path/to/clean-dataset.csv \
                                    --n-buckets 10 \
                                    --window-size 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
def find_diverse(csv_path: str, n_buckets: int, window_size: int) -> list[dict]:
    """
    Return a list of {window_index, patient_id, glucose_mean, glucose_bucket}
    dicts, one per requested bucket, spanning the full glucose range.
    """
    df      = pd.read_csv(csv_path)
    n_rows  = len(df)
    n_wins  = n_rows // window_size
    logger.info("Dataset: %d rows → %d windows  (window_size=%d)", n_rows, n_wins, window_size)

    # Build window-level summary
    records = []
    for i in range(n_wins):
        chunk     = df.iloc[i * window_size : (i + 1) * window_size]
        glc_mean  = float(chunk["Glucose_level"].mean())
        glc_std   = float(chunk["Glucose_level"].std())
        pid       = int(chunk["Patient_Id"].iloc[0])
        ppg_vals  = chunk["PPG_Signal"].values.astype(np.float32)
        ppg_std   = float(np.std(ppg_vals))
        records.append({
            "window_index": i,
            "patient_id":   pid,
            "glucose_mean": glc_mean,
            "glucose_std":  glc_std,
            "ppg_std":      ppg_std,
        })

    all_glucoses = [r["glucose_mean"] for r in records]
    g_min, g_max = min(all_glucoses), max(all_glucoses)
    logger.info("Glucose range: %.1f – %.1f mg/dL", g_min, g_max)

    # Divide range into equal-width buckets and pick one window per bucket
    bins       = np.linspace(g_min, g_max, n_buckets + 1)
    selected   = []
    used_wins  = set()

    for lo, hi in zip(bins[:-1], bins[1:]):
        candidates = [
            r for r in records
            if lo <= r["glucose_mean"] < hi and r["window_index"] not in used_wins
        ]
        if not candidates:
            logger.debug("  Bucket [%.1f, %.1f): no candidates", lo, hi)
            continue
        # Prefer windows with variable PPG signal (std > 1) to avoid flat signals
        variable_cands = [c for c in candidates if c["ppg_std"] > 1.0]
        pool = variable_cands if variable_cands else candidates
        best = max(pool, key=lambda r: r["ppg_std"])
        best["glucose_bucket"] = f"{lo:.1f}–{hi:.1f}"
        selected.append(best)
        used_wins.add(best["window_index"])
        logger.info(
            "  Bucket %-14s → window %4d  patient %2d  glucose %.1f  ppg_std %.2f",
            best["glucose_bucket"],
            best["window_index"],
            best["patient_id"],
            best["glucose_mean"],
            best["ppg_std"],
        )

    return selected


# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find diverse glucose windows for visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",         default=str(ROOT / "Archive_Dataset" / "clean-dataset.csv"))
    p.add_argument("--n-buckets",   type=int, default=8,   metavar="N")
    p.add_argument("--window-size", type=int, default=500, metavar="W")
    p.add_argument("--out",         default=str(ROOT / "diverse_samples.json"))
    return p.parse_args()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = _parse()

    if not Path(args.csv).exists():
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)

    samples = find_diverse(args.csv, args.n_buckets, args.window_size)

    if not samples:
        logger.error("No diverse samples found.")
        sys.exit(1)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)
    logger.info("\nSaved %d diverse samples to: %s", len(samples), out_path)

    indices = [s["window_index"] for s in samples]
    logger.info("Selected window indices: %s", indices)
