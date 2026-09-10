"""
prepare_generalization_data.py
==============================
Align PhysioNet BIG IDEAs BVP + Dexcom data and prepare resampled windows
compatible with the trained MultimodalFusionNet (expects 500-sample input).

Dataset specifics:
  - BVP (Empatica E4): 64 Hz, column 'datetime', column ' bvp' (leading space)
  - Dexcom G6 CGM    : ~5-min intervals, column 'Timestamp (YYYY-MM-DDThh:mm:ss)'
                       column 'Glucose Value (mg/dL)', metadata rows present

Output:
  physionet_X_500.npy   — shape (N, 500), float32, resampled PPG windows
  physionet_y_glucose.npy — shape (N,),   float32, mg/dL glucose labels

Usage:
  python prepare_generalization_data.py
  python prepare_generalization_data.py --data-dir ~/Desktop/physionet.org/.../001
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BVP_HZ          = 64            # Empatica E4 BVP sample rate
WINDOW_SEC      = 5             # seconds of BVP to extract per CGM reading
BVP_SAMPLES     = BVP_HZ * WINDOW_SEC   # = 320 samples at 64 Hz
TARGET_SAMPLES  = 500           # model's expected input length (100 Hz × 5 s)
GLUCOSE_MIN     = 40
GLUCOSE_MAX     = 400


def parse_dexcom(path: Path) -> pd.DataFrame:
    """
    Load Dexcom CSV and return EGV glucose rows with valid numeric glucose.

    The Dexcom export mixes metadata rows (FirstName, LastName, Alert, etc.)
    with EGV (Estimated Glucose Value) rows. We keep only EGV rows with a
    non-NaN glucose value in 'Glucose Value (mg/dL)'.
    """
    log.info("Loading Dexcom: %s", path)
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names
    df.columns = df.columns.str.strip()

    ts_col  = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    glc_col = "Glucose Value (mg/dL)"

    if ts_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{ts_col}' not found. Found: {list(df.columns)}"
        )
    if glc_col not in df.columns:
        raise KeyError(
            f"Glucose column '{glc_col}' not found. Found: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["glucose"]   = pd.to_numeric(df[glc_col], errors="coerce")

    # Drop metadata rows (no valid timestamp or glucose)
    df = df.dropna(subset=["timestamp", "glucose"])
    df = df[(df["glucose"] >= GLUCOSE_MIN) & (df["glucose"] <= GLUCOSE_MAX)]
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info(
        "  Dexcom: %d valid EGV readings | glucose %.1f – %.1f mg/dL | "
        "span: %s → %s",
        len(df),
        df["glucose"].min(), df["glucose"].max(),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
    )
    return df[["timestamp", "glucose"]]


def load_bvp(path: Path) -> pd.Series:
    """
    Load BVP CSV into a pandas Series indexed by datetime.

    The file is 1.3 GB / 40 M rows — loaded once into memory (≈600 MB).
    Column names may have leading/trailing spaces; we strip them.
    """
    log.info("Loading BVP: %s  (1.3 GB — may take ~30 s) ...", path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()   # removes leading space from ' bvp'

    ts_col  = "datetime"
    bvp_col = "bvp"

    if ts_col not in df.columns:
        raise KeyError(
            f"Time column '{ts_col}' not found. Found: {list(df.columns)}"
        )
    if bvp_col not in df.columns:
        raise KeyError(
            f"BVP column '{bvp_col}' not found after strip. Found: {list(df.columns)}"
        )

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df[bvp_col] = pd.to_numeric(df[bvp_col], errors="coerce")
    df = df.dropna(subset=[ts_col, bvp_col])

    series = df.set_index(ts_col)[bvp_col].sort_index()
    log.info(
        "  BVP: %d samples | span: %s → %s",
        len(series),
        series.index[0],
        series.index[-1],
    )
    return series


def extract_and_resample(
    bvp: pd.Series,
    cgm_df: pd.DataFrame,
    window_sec: int = WINDOW_SEC,
    bvp_hz: int     = BVP_HZ,
    target_samples: int = TARGET_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each CGM timestamp extract the immediately preceding `window_sec`-second
    BVP window (320 samples at 64 Hz) and resample to `target_samples` (500).

    Returns:
        X : np.ndarray, shape (N, target_samples), float32
        y : np.ndarray, shape (N,),                float32
    """
    window_dur = pd.Timedelta(seconds=window_sec)
    min_samples = int(bvp_hz * window_sec * 0.9)  # allow 10% dropout tolerance

    X_list, y_list = [], []
    skipped = 0

    for i, row in cgm_df.iterrows():
        t_end   = row["timestamp"]
        t_start = t_end - window_dur
        glucose = float(row["glucose"])

        # Slice using sorted DatetimeIndex (binary search internally)
        window = bvp[t_start : t_end]

        if len(window) < min_samples:
            skipped += 1
            continue

        # Use the last `bvp_samples` points to stay exactly at the CGM timestamp
        raw = window.values[-int(bvp_hz * window_sec):]
        if len(raw) < min_samples:
            skipped += 1
            continue

        # ── Resample 64 Hz → 100 Hz (320 → 500 samples) ──────────────────
        resampled = resample(raw.astype(np.float32), target_samples)

        X_list.append(resampled.astype(np.float32))
        y_list.append(glucose)

    log.info(
        "  Extracted %d windows | skipped %d (insufficient BVP coverage)",
        len(X_list), skipped,
    )

    if not X_list:
        raise RuntimeError(
            "No windows were extracted. Check that BVP and Dexcom timestamps overlap."
        )

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Align PhysioNet BIG IDEAs BVP+Dexcom → resampled windows"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to subject folder (default: auto-detect 001/)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to save .npy files (default: same folder as this script)",
    )
    args = parser.parse_args()

    # ── Locate subject folder ─────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
    else:
        candidates = [
            Path.home() / "Desktop" / "physionet.org" / "files"
            / "big-ideas-glycemic-wearable" / "1.1.3" / "001",
            script_dir / "physionet.org" / "files"
            / "big-ideas-glycemic-wearable" / "1.1.3" / "001",
        ]
        data_dir = next((c for c in candidates if c.exists()), None)

    if data_dir is None or not data_dir.exists():
        log.error(
            "Cannot find data directory. Pass --data-dir explicitly.\n"
            "Expected path: ~/Desktop/physionet.org/files/"
            "big-ideas-glycemic-wearable/1.1.3/001"
        )
        sys.exit(1)

    log.info("Data directory: %s", data_dir)

    bvp_path    = data_dir / "BVP_001.csv"
    dexcom_path = data_dir / "Dexcom_001.csv"

    for p in (bvp_path, dexcom_path):
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    # ── Parse & align ─────────────────────────────────────────────────────────
    cgm_df = parse_dexcom(dexcom_path)
    bvp    = load_bvp(bvp_path)

    X, y = extract_and_resample(bvp, cgm_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else script_dir
    x_path  = out_dir / "physionet_X_500.npy"
    y_path  = out_dir / "physionet_y_glucose.npy"

    np.save(x_path, X)
    np.save(y_path, y)

    log.info("Saved: %s  shape=%s", x_path, X.shape)
    log.info("Saved: %s  shape=%s", y_path, y.shape)
    log.info(
        "\nSummary:"
        "\n  Windows (N):          %d"
        "\n  Samples per window:   %d  (resampled from 320→500)"
        "\n  Glucose range:        %.1f – %.1f mg/dL"
        "\n  Glucose mean ± std:   %.1f ± %.1f mg/dL",
        len(X), X.shape[1],
        y.min(), y.max(),
        y.mean(), y.std(),
    )


if __name__ == "__main__":
    main()
