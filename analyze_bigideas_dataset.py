"""
analyze_bigideas_dataset.py
===========================
Comprehensive analysis of the BIG IDEAs Lab Glycemic Variability & Wearable
Device Dataset (PhysioNet, v1.1.3, doi:10.13026/aw6y-fc44).

For each downloaded subject folder this script:
  1. Parses Dexcom G6 CGM data (EGV rows only, ~5-min intervals)
  2. Parses BVP (PPG at 64 Hz) in chunked streaming mode to avoid OOM
  3. Temporally aligns BVP windows with CGM labels
  4. Computes signal quality index (SQI) for each window
  5. Generates window-glucose pairs comparable to our archive dataset
  6. Produces publication-quality figures and a JSON summary

Outputs
-------
  BigIdeas_Analysis_Report.json
  BigIdeas_Fig1_Glucose_Timeseries.png
  BigIdeas_Fig2_BVP_Sample.png
  BigIdeas_Fig3_Alignment_Timeline.png
  BigIdeas_Fig4_Glucose_Distribution.png
  BigIdeas_Fig5_Dataset_Comparison.png
  bigideas_windows.csv        (window metadata: subject, t_start, glucose, sqi)

Usage
-----
  python analyze_bigideas_dataset.py
  python analyze_bigideas_dataset.py --data-dir /path/to/big-ideas-glycemic-wearable/1.1.3
  python analyze_bigideas_dataset.py --max-subjects 3
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BVP_HZ          = 64          # Empatica E4 PPG sampling rate
WINDOW_SAMPLES  = 500         # same as archive dataset
WINDOW_SEC      = WINDOW_SAMPLES / BVP_HZ   # ≈7.8 s per window
CGM_INTERVAL_S  = 300         # Dexcom G6: 5-minute readings
# Windows to extract per CGM reading (stride = WINDOW_SAMPLES, no overlap)
WINDOWS_PER_CGM = int(CGM_INTERVAL_S * BVP_HZ / WINDOW_SAMPLES)  # = 38
BVP_CHUNK_ROWS  = 2_000_000   # stream BVP in ~31-second chunks to save memory
GLUCOSE_MIN     = 40
GLUCOSE_MAX     = 400

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Dexcom parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_dexcom(path: Path) -> pd.DataFrame:
    """Return EGV glucose readings as DataFrame with columns [timestamp, glucose]."""
    df = pd.read_csv(path, low_memory=False)

    # Find the EGV rows — 'Event Type' column contains 'EGV'
    # Column names vary slightly; try both 'Event Type' and 'Timestamp (YYYY-MM-DDThh:mm:ss)'
    event_col = None
    for c in df.columns:
        if "event" in c.lower() and "type" in c.lower():
            event_col = c
            break

    ts_col = None
    for c in df.columns:
        if "timestamp" in c.lower():
            ts_col = c
            break

    glucose_col = None
    for c in df.columns:
        if "glucose" in c.lower() and "value" in c.lower():
            glucose_col = c
            break

    if event_col is None or ts_col is None or glucose_col is None:
        raise ValueError(
            f"Could not identify required columns in {path}. "
            f"Found: {list(df.columns)}"
        )

    egv = df[df[event_col].astype(str).str.strip().str.upper() == "EGV"].copy()
    egv = egv[[ts_col, glucose_col]].rename(
        columns={ts_col: "timestamp", glucose_col: "glucose"}
    )

    # Parse timestamps — Dexcom uses both 'YYYY-MM-DDThh:mm:ss' and 'YYYY-MM-DD hh:mm:ss'
    egv["timestamp"] = pd.to_datetime(egv["timestamp"], errors="coerce")
    egv["glucose"] = pd.to_numeric(egv["glucose"], errors="coerce")

    # Drop rows with 'Low' / 'High' text (non-numeric) and out-of-range
    egv = egv.dropna(subset=["timestamp", "glucose"])
    egv = egv[(egv["glucose"] >= GLUCOSE_MIN) & (egv["glucose"] <= GLUCOSE_MAX)]
    egv = egv.sort_values("timestamp").reset_index(drop=True)

    log.info("    Dexcom: %d EGV readings  |  glucose %.1f–%.1f mg/dL",
             len(egv), egv["glucose"].min(), egv["glucose"].max())
    return egv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: BVP parser (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_bvp_columns(path: Path):
    """Return (timestamp_col, value_col) names by peeking at header."""
    header = pd.read_csv(path, nrows=1)
    cols = list(header.columns)
    ts_col  = next((c for c in cols if "time" in c.lower() or "date" in c.lower()), cols[0])
    val_col = next((c for c in cols if "bvp" in c.lower() or "value" in c.lower()), cols[1])
    return ts_col, val_col


def stream_bvp_windows(
    path: Path,
    cgm_df: pd.DataFrame,
    subject_id: str,
) -> pd.DataFrame:
    """
    Stream the BVP CSV and extract one-window-per-CGM-reading records.

    For each CGM timestamp t_cgm, we extract the WINDOWS_PER_CGM non-overlapping
    windows of WINDOW_SAMPLES that fall within [t_cgm - CGM_INTERVAL_S/2, t_cgm].
    Each window gets the glucose label from that CGM reading.

    Returns a DataFrame with one row per window:
      subject, t_start, glucose, sqi, mean_bvp, std_bvp, n_samples
    """
    ts_col, val_col = _detect_bvp_columns(path)
    log.info("    BVP columns: ts='%s'  val='%s'", ts_col, val_col)

    # Build an index of (t_cgm ± half_interval) for fast lookup
    half = pd.Timedelta(seconds=CGM_INTERVAL_S / 2)

    # We'll load BVP in chunks and accumulate windows into a buffer
    results = []
    buffer_ts   = []   # list of timestamps (datetime)
    buffer_vals = []   # list of float values

    def _flush_buffer_against_cgm():
        """Check buffer for complete windows aligned to CGM readings."""
        if len(buffer_ts) < WINDOW_SAMPLES:
            return
        buf_arr_ts  = np.array(buffer_ts,   dtype="datetime64[ms]")
        buf_arr_val = np.array(buffer_vals, dtype=np.float32)

        for _, cgm_row in cgm_df.iterrows():
            t_cgm  = np.datetime64(cgm_row["timestamp"])
            t_low  = t_cgm - np.timedelta64(int(CGM_INTERVAL_S * 1000), "ms")
            t_high = t_cgm

            mask = (buf_arr_ts >= t_low) & (buf_arr_ts <= t_high)
            idx_in_window = np.where(mask)[0]

            if len(idx_in_window) < WINDOW_SAMPLES:
                continue

            # Slice into non-overlapping sub-windows
            for start_offset in range(0, len(idx_in_window) - WINDOW_SAMPLES + 1,
                                      WINDOW_SAMPLES):
                w_idx = idx_in_window[start_offset: start_offset + WINDOW_SAMPLES]
                signal = buf_arr_val[w_idx]
                sqi    = float(np.std(signal))   # simple std-based SQI
                results.append({
                    "subject":   subject_id,
                    "t_start":   str(buf_arr_ts[w_idx[0]]),
                    "glucose":   float(cgm_row["glucose"]),
                    "sqi":       sqi,
                    "mean_bvp":  float(np.mean(signal)),
                    "std_bvp":   sqi,
                    "n_samples": WINDOW_SAMPLES,
                })

    total_rows = 0
    for chunk in pd.read_csv(
        path,
        chunksize=BVP_CHUNK_ROWS,
        usecols=[ts_col, val_col],
        low_memory=True,
    ):
        chunk = chunk.rename(columns={ts_col: "ts", val_col: "val"})
        chunk["ts"] = pd.to_datetime(chunk["ts"], errors="coerce")
        chunk = chunk.dropna(subset=["ts", "val"])
        chunk["val"] = pd.to_numeric(chunk["val"], errors="coerce")
        chunk = chunk.dropna(subset=["val"])

        buffer_ts.extend(chunk["ts"].values.tolist())
        buffer_vals.extend(chunk["val"].values.tolist())
        total_rows += len(chunk)

        # Keep buffer manageable: only retain last 3x CGM interval of data
        keep_samples = 3 * CGM_INTERVAL_S * BVP_HZ
        if len(buffer_ts) > keep_samples:
            _flush_buffer_against_cgm()
            buffer_ts   = buffer_ts[-keep_samples:]
            buffer_vals = buffer_vals[-keep_samples:]

        if total_rows % 5_000_000 == 0:
            log.info("      Streamed %d M BVP rows, %d windows so far ...",
                     total_rows // 1_000_000, len(results))

    # Final flush
    _flush_buffer_against_cgm()
    log.info("    BVP: %d total rows streamed, %d windows extracted",
             total_rows, len(results))

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).drop_duplicates(subset=["subject", "t_start"])


# ─────────────────────────────────────────────────────────────────────────────
# Signal quality statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sqi_threshold(windows_df: pd.DataFrame) -> float:
    """5th percentile of non-zero STDs — mirrors archive dataset logic."""
    nonzero = windows_df["sqi"][windows_df["sqi"] > 1e-6]
    if len(nonzero) == 0:
        return 1e-4
    return float(np.percentile(nonzero, 5))


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_glucose_timeseries(cgm_df: pd.DataFrame, subject_id: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(cgm_df["timestamp"], cgm_df["glucose"],
            color="#2196F3", linewidth=1.0, label="CGM (Dexcom G6)")
    ax.fill_between(cgm_df["timestamp"], cgm_df["glucose"],
                    alpha=0.15, color="#2196F3")
    ax.axhline(70,  color="green",  linestyle="--", linewidth=0.8, label="Hypoglycemia (70 mg/dL)")
    ax.axhline(180, color="orange", linestyle="--", linewidth=0.8, label="Hyperglycemia (180 mg/dL)")
    ax.set_xlabel("Date/Time", fontsize=11)
    ax.set_ylabel("Interstitial Glucose (mg/dL)", fontsize=11)
    ax.set_title(f"Continuous Glucose Monitor — Subject {subject_id} "
                 f"({len(cgm_df)} readings over "
                 f"{(cgm_df['timestamp'].iloc[-1]-cgm_df['timestamp'].iloc[0]).days} days)",
                 fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    stats_text = (
        f"Mean: {cgm_df['glucose'].mean():.1f} | "
        f"Std: {cgm_df['glucose'].std():.1f} | "
        f"Min: {cgm_df['glucose'].min():.0f} | "
        f"Max: {cgm_df['glucose'].max():.0f} mg/dL"
    )
    ax.set_title(ax.get_title() + f"\n{stats_text}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("    Saved: %s", out_path)


def fig_bvp_sample(bvp_sample: np.ndarray, fs: int, subject_id: str, glucose: float,
                   out_path: Path):
    t = np.arange(len(bvp_sample)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Time domain
    axes[0].plot(t, bvp_sample, color="#E53935", linewidth=0.8)
    axes[0].set_xlabel("Time (s)", fontsize=10)
    axes[0].set_ylabel("BVP (a.u.)", fontsize=10)
    axes[0].set_title(f"BVP/PPG Sample Window — Subject {subject_id}  "
                      f"(glucose ≈ {glucose:.0f} mg/dL, {len(bvp_sample)/fs:.1f}s @ {fs}Hz)",
                      fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Frequency spectrum
    n   = len(bvp_sample)
    fft = np.abs(np.fft.rfft(bvp_sample - bvp_sample.mean()))
    freqs = np.fft.rfftfreq(n, d=1/fs)
    hr_mask = (freqs >= 0.5) & (freqs <= 4.0)   # 30–240 bpm
    axes[1].plot(freqs[hr_mask], fft[hr_mask], color="#7B1FA2", linewidth=0.9)
    axes[1].set_xlabel("Frequency (Hz)", fontsize=10)
    axes[1].set_ylabel("Amplitude (a.u.)", fontsize=10)
    axes[1].set_title("Frequency Spectrum (0.5–4 Hz cardiac band)", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("    Saved: %s", out_path)


def fig_alignment_timeline(cgm_df: pd.DataFrame, windows_df: pd.DataFrame,
                           subject_id: str, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # CGM glucose trace
    axes[0].plot(cgm_df["timestamp"], cgm_df["glucose"],
                 color="#2196F3", linewidth=1.2, label="CGM")
    axes[0].set_ylabel("Glucose (mg/dL)", fontsize=10)
    axes[0].set_title(f"BVP–CGM Temporal Alignment — Subject {subject_id}", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Window coverage
    if not windows_df.empty:
        win_times = pd.to_datetime(windows_df["t_start"])
        axes[1].scatter(win_times, windows_df["sqi"],
                        s=2, alpha=0.4, color="#43A047", label="Extracted windows")
        sqi_thresh = compute_sqi_threshold(windows_df)
        axes[1].axhline(sqi_thresh, color="red", linestyle="--", linewidth=0.8,
                        label=f"SQI threshold ({sqi_thresh:.3f})")
        axes[1].set_ylabel("SQI (std of BVP)", fontsize=10)
        axes[1].set_xlabel("Date/Time", fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No windows extracted",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("    Saved: %s", out_path)


def fig_glucose_distribution(windows_df: pd.DataFrame,
                              archive_path: Path | None,
                              out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gluc = windows_df["glucose"].dropna()
    axes[0].hist(gluc, bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0].axvline(gluc.mean(), color="red", linewidth=1.5,
                    label=f"Mean: {gluc.mean():.1f}")
    axes[0].axvline(70,  color="green",  linestyle="--", linewidth=0.8, label="Hypo (70)")
    axes[0].axvline(180, color="orange", linestyle="--", linewidth=0.8, label="Hyper (180)")
    axes[0].set_xlabel("Glucose (mg/dL)", fontsize=11)
    axes[0].set_ylabel("Window count", fontsize=11)
    axes[0].set_title(f"BigIdeas: Glucose Distribution\n"
                      f"N={len(gluc)} windows | std={gluc.std():.1f} mg/dL", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Comparison: BigIdeas vs Archive
    if archive_path and archive_path.exists():
        try:
            archive = pd.read_csv(archive_path, usecols=["Glucose_level"])
            ax2 = axes[1]
            ax2.hist(archive["Glucose_level"].dropna(), bins=40,
                     color="#FF7043", alpha=0.7, label="Archive (ours)", edgecolor="white")
            ax2.hist(gluc, bins=40, color="#2196F3", alpha=0.6,
                     label="BigIdeas", edgecolor="white")
            ax2.set_xlabel("Glucose (mg/dL)", fontsize=11)
            ax2.set_ylabel("Window count", fontsize=11)
            ax2.set_title("Dataset Glucose Distribution Comparison", fontsize=11)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            log.info("    Archive glucose: mean=%.1f, std=%.1f",
                     archive["Glucose_level"].mean(), archive["Glucose_level"].std())
        except Exception as e:
            log.warning("    Could not load archive dataset: %s", e)
            axes[1].text(0.5, 0.5, "Archive dataset not available",
                         ha="center", va="center", transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "Archive dataset not found\n(expected: Archive_Dataset/clean-dataset.csv)",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("    Saved: %s", out_path)


def fig_within_patient_variability(windows_df: pd.DataFrame,
                                   archive_path: Path | None,
                                   out_path: Path):
    """
    Key figure: within-patient glucose variability.
    Shows why BigIdeas is a better evaluation set — glucose actually changes.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Glucose over time (BigIdeas, by subject) ──────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    if not windows_df.empty:
        for subj, grp in windows_df.groupby("subject"):
            times = pd.to_datetime(grp["t_start"])
            ax_a.scatter(times, grp["glucose"], s=3, alpha=0.5, label=str(subj))
        ax_a.set_xlabel("Time", fontsize=10)
        ax_a.set_ylabel("Glucose (mg/dL)", fontsize=10)
        ax_a.set_title("BigIdeas: Within-Patient Glucose Variation (Window Level)",
                        fontsize=11)
        ax_a.grid(True, alpha=0.3)
        fig.autofmt_xdate()

    # ── Panel B: Glucose std per subject ───────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    if not windows_df.empty:
        per_subj = windows_df.groupby("subject")["glucose"].std().reset_index()
        ax_b.bar(per_subj["subject"].astype(str), per_subj["glucose"],
                 color="#2196F3", alpha=0.8)
        ax_b.set_xlabel("Subject", fontsize=10)
        ax_b.set_ylabel("Within-Subject Glucose Std (mg/dL)", fontsize=10)
        ax_b.set_title("Intra-Subject Glucose Variability", fontsize=11)
        ax_b.grid(True, alpha=0.3, axis="y")

    # ── Panel C: SQI distribution ──────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    if not windows_df.empty:
        sqi_thresh = compute_sqi_threshold(windows_df)
        ax_c.hist(windows_df["sqi"], bins=50, color="#7B1FA2", alpha=0.8,
                  edgecolor="white")
        ax_c.axvline(sqi_thresh, color="red", linestyle="--", linewidth=1.2,
                     label=f"Threshold: {sqi_thresh:.3f}")
        n_clean = (windows_df["sqi"] >= sqi_thresh).sum()
        ax_c.set_xlabel("SQI (BVP std, ADC)", fontsize=10)
        ax_c.set_ylabel("Window count", fontsize=10)
        ax_c.set_title(f"Signal Quality: {n_clean}/{len(windows_df)} "
                       f"({n_clean/len(windows_df)*100:.0f}%) clean",
                       fontsize=11)
        ax_c.legend(fontsize=9)
        ax_c.grid(True, alpha=0.3)

    # ── Panel D: Dataset comparison table ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1:])
    ax_d.axis("off")

    n_windows = len(windows_df)
    n_subjects = windows_df["subject"].nunique() if not windows_df.empty else 0
    gluc = windows_df["glucose"] if not windows_df.empty else pd.Series([])

    table_data = [
        ["Metric",               "Archive Dataset",    "BigIdeas (downloaded)"],
        ["Subjects",             "22",                 str(n_subjects)],
        ["Total windows",        "1,689",              str(n_windows)],
        ["Glucose range (mg/dL)","~70–200 (est.)",     f"{gluc.min():.0f}–{gluc.max():.0f}" if len(gluc) else "—"],
        ["Glucose std (mg/dL)",  "≈0 within patient",  f"{gluc.std():.1f}" if len(gluc) else "—"],
        ["PPG sample rate",      "Unknown (archive)",  "64 Hz (Empatica E4)"],
        ["CGM interval",         "Per-patient label",  "5 min (Dexcom G6)"],
        ["Recording duration",   "~10 days (est.)",    "8–10 days"],
        ["Within-patient dyn.",  "NO (limitation)",    "YES (key upgrade)"],
    ]

    table = ax_d.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:
            cell.set_facecolor("#E3F2FD")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    ax_d.set_title("Dataset Comparison: Archive vs BigIdeas", fontsize=11, pad=10)

    plt.suptitle("BigIdeas Glycemic Variability Dataset — Full Analysis", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("    Saved: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_subject(subject_dir: Path, out_dir: Path) -> dict:
    subject_id = subject_dir.name
    log.info("\n  ── Subject %s ──", subject_id)

    bvp_path   = subject_dir / f"BVP_{subject_id}.csv"
    dexcom_path = subject_dir / f"Dexcom_{subject_id}.csv"

    if not bvp_path.exists():
        log.warning("    BVP file not found: %s — skipping", bvp_path)
        return {}
    if not dexcom_path.exists():
        log.warning("    Dexcom file not found: %s — skipping", dexcom_path)
        return {}

    # 1. Parse Dexcom
    cgm_df = parse_dexcom(dexcom_path)
    if cgm_df.empty:
        log.warning("    No valid CGM readings found — skipping")
        return {}

    cgm_stats = {
        "n_readings":     int(len(cgm_df)),
        "duration_days":  float((cgm_df["timestamp"].iloc[-1] - cgm_df["timestamp"].iloc[0])
                                / pd.Timedelta(days=1)),
        "glucose_mean":   float(cgm_df["glucose"].mean()),
        "glucose_std":    float(cgm_df["glucose"].std()),
        "glucose_min":    float(cgm_df["glucose"].min()),
        "glucose_max":    float(cgm_df["glucose"].max()),
        "glucose_cv":     float(cgm_df["glucose"].std() / cgm_df["glucose"].mean() * 100),
        "time_in_normal": float(
            ((cgm_df["glucose"] >= 70) & (cgm_df["glucose"] <= 180)).mean() * 100
        ),
        "time_hypo":      float((cgm_df["glucose"] < 70).mean() * 100),
        "time_hyper":     float((cgm_df["glucose"] > 180).mean() * 100),
    }

    # 2. Generate glucose timeseries figure
    fig_glucose_timeseries(
        cgm_df, subject_id,
        out_dir / f"BigIdeas_{subject_id}_Glucose_Timeseries.png"
    )

    # 3. Stream BVP and extract windows
    log.info("    Streaming BVP (this may take 1–3 min for large files) ...")
    windows_df = stream_bvp_windows(bvp_path, cgm_df, subject_id)

    if windows_df.empty:
        log.warning("    No aligned windows extracted for subject %s", subject_id)
        bvp_stats  = {"status": "no_windows"}
        win_stats  = {}
    else:
        # 4. Signal quality
        sqi_thresh = compute_sqi_threshold(windows_df)
        clean_mask = windows_df["sqi"] >= sqi_thresh
        log.info("    SQI threshold: %.4f  |  clean windows: %d/%d (%.1f%%)",
                 sqi_thresh, clean_mask.sum(), len(windows_df),
                 clean_mask.mean() * 100)

        bvp_stats = {
            "n_windows_total":   int(len(windows_df)),
            "n_windows_clean":   int(clean_mask.sum()),
            "pct_clean":         float(clean_mask.mean() * 100),
            "sqi_threshold":     float(sqi_thresh),
            "bvp_mean_sqi":      float(windows_df["sqi"].mean()),
        }
        win_stats = {
            "glucose_mean_windowed": float(windows_df["glucose"].mean()),
            "glucose_std_windowed":  float(windows_df["glucose"].std()),
            "glucose_min_windowed":  float(windows_df["glucose"].min()),
            "glucose_max_windowed":  float(windows_df["glucose"].max()),
        }

        # 5. BVP sample figure (pick a window with reasonable SQI)
        best_windows = windows_df[clean_mask].sort_values("sqi", ascending=False)
        if not best_windows.empty:
            # Reload just the first good window's BVP for the plot
            best_t = pd.Timestamp(best_windows.iloc[0]["t_start"])
            best_glucose = best_windows.iloc[0]["glucose"]

            # Read only the rows around best_t
            ts_col, val_col = _detect_bvp_columns(bvp_path)
            half_sec = CGM_INTERVAL_S
            rows_to_read = half_sec * BVP_HZ + WINDOW_SAMPLES * 2
            sample_chunk = None
            for chunk in pd.read_csv(
                bvp_path, chunksize=rows_to_read * 2,
                usecols=[ts_col, val_col], low_memory=True
            ):
                chunk = chunk.rename(columns={ts_col: "ts", val_col: "val"})
                chunk["ts"] = pd.to_datetime(chunk["ts"], errors="coerce")
                near = chunk[
                    (chunk["ts"] >= best_t) & (chunk["ts"] <= best_t + pd.Timedelta(seconds=WINDOW_SEC + 1))
                ]
                if len(near) >= WINDOW_SAMPLES:
                    sample_chunk = near.iloc[:WINDOW_SAMPLES]["val"].values.astype(float)
                    break
            if sample_chunk is not None and len(sample_chunk) == WINDOW_SAMPLES:
                fig_bvp_sample(
                    sample_chunk, BVP_HZ, subject_id, best_glucose,
                    out_dir / f"BigIdeas_{subject_id}_BVP_Sample.png"
                )

        # 6. Alignment timeline figure
        fig_alignment_timeline(
            cgm_df, windows_df, subject_id,
            out_dir / f"BigIdeas_{subject_id}_Alignment_Timeline.png"
        )

    return {
        "subject_id":  subject_id,
        "cgm":         cgm_stats,
        "bvp":         bvp_stats,
        "windows":     win_stats,
        "windows_df":  windows_df,  # kept in memory for combined figures
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BigIdeas glycemic wearable dataset analyser")
    parser.add_argument(
        "--data-dir", default=None,
        help="Root folder of the downloaded dataset (default: auto-detect)"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output folder for figures and reports (default: same as script)"
    )
    parser.add_argument(
        "--max-subjects", type=int, default=99,
        help="Limit analysis to first N subjects (useful for a quick test)"
    )
    args = parser.parse_args()

    # ── Locate dataset root ───────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    candidate_dirs = [
        Path(args.data_dir) if args.data_dir else None,
        script_dir / "physionet.org" / "files" / "big-ideas-glycemic-wearable" / "1.1.3",
        script_dir / "big-ideas-glycemic-wearable" / "1.1.3",
        script_dir / "bigideas",
    ]
    data_dir = None
    for cd in candidate_dirs:
        if cd and cd.exists() and any(cd.iterdir()):
            data_dir = cd
            break

    if data_dir is None:
        log.error("Cannot find BigIdeas dataset directory. "
                  "Place data at: %s",
                  script_dir / "physionet.org/files/big-ideas-glycemic-wearable/1.1.3")
        sys.exit(1)

    log.info("Dataset root: %s", data_dir)

    out_dir = Path(args.out_dir) if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Demographics ──────────────────────────────────────────────────────────
    demo_path = data_dir / "Demographics.csv"
    demographics = {}
    if demo_path.exists():
        demo_df = pd.read_csv(demo_path)
        log.info("Demographics: %d subjects | HbA1c %.1f–%.1f%%",
                 len(demo_df), demo_df["HbA1c"].min(), demo_df["HbA1c"].max())
        demographics = demo_df.to_dict(orient="records")

    # ── Discover subject folders ──────────────────────────────────────────────
    subject_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )[: args.max_subjects]

    log.info("Found %d subject folder(s) to analyse: %s",
             len(subject_dirs), [d.name for d in subject_dirs])

    # ── Per-subject analysis ──────────────────────────────────────────────────
    all_results = []
    all_windows: list[pd.DataFrame] = []

    for subj_dir in subject_dirs:
        result = analyze_subject(subj_dir, out_dir)
        if result:
            all_results.append({k: v for k, v in result.items() if k != "windows_df"})
            if not result.get("windows_df", pd.DataFrame()).empty:
                all_windows.append(result["windows_df"])

    # ── Combined figures ──────────────────────────────────────────────────────
    if all_windows:
        combined_df = pd.concat(all_windows, ignore_index=True)
        log.info("\nCombined: %d windows across %d subject(s)",
                 len(combined_df), combined_df["subject"].nunique())

        archive_csv = script_dir / "Archive_Dataset" / "clean-dataset.csv"

        fig_glucose_distribution(
            combined_df, archive_csv,
            out_dir / "BigIdeas_Fig_Glucose_Distribution.png"
        )
        fig_within_patient_variability(
            combined_df, archive_csv,
            out_dir / "BigIdeas_Fig_Full_Analysis.png"
        )

        # Save window metadata CSV
        combined_df.to_csv(out_dir / "bigideas_windows.csv", index=False)
        log.info("Saved: bigideas_windows.csv (%d rows)", len(combined_df))
    else:
        combined_df = pd.DataFrame()
        log.warning("No windows extracted — skipping combined figures")

    # ── Summary report ────────────────────────────────────────────────────────
    gluc = combined_df["glucose"] if not combined_df.empty else pd.Series([])
    report = {
        "dataset":      "BIG IDEAs Lab Glycemic Variability & Wearable Device Dataset v1.1.3",
        "doi":          "10.13026/aw6y-fc44",
        "bvp_hz":       BVP_HZ,
        "cgm_interval_min": CGM_INTERVAL_S // 60,
        "window_samples": WINDOW_SAMPLES,
        "window_sec":   round(WINDOW_SEC, 2),
        "downloaded_subjects": len(subject_dirs),
        "total_subjects_in_dataset": 16,
        "demographics": demographics,
        "total_windows": int(len(combined_df)),
        "total_subjects_with_windows": int(combined_df["subject"].nunique()) if not combined_df.empty else 0,
        "glucose_stats": {
            "mean":  float(gluc.mean()) if len(gluc) else None,
            "std":   float(gluc.std())  if len(gluc) else None,
            "min":   float(gluc.min())  if len(gluc) else None,
            "max":   float(gluc.max())  if len(gluc) else None,
            "cv_pct": float(gluc.std() / gluc.mean() * 100) if len(gluc) else None,
        },
        "comparison_with_archive": {
            "archive_within_patient_glucose_std": "≈0 mg/dL (label essentially constant per patient)",
            "bigideas_within_patient_glucose_std": f"{gluc.std():.1f} mg/dL" if len(gluc) else "N/A",
            "advantage": "BigIdeas provides true within-patient glucose dynamics — "
                         "directly addresses the primary limitation of our archive dataset",
        },
        "per_subject": all_results,
    }

    report_path = out_dir / "BigIdeas_Analysis_Report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Saved: %s", report_path)

    # ── Console summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  BigIdeas Dataset Analysis — Summary")
    log.info("=" * 60)
    log.info("  Subjects analysed:    %d / 16 downloaded", len(subject_dirs))
    log.info("  Total BVP windows:    %d", len(combined_df))
    if not combined_df.empty:
        sqi_thresh = compute_sqi_threshold(combined_df)
        n_clean = (combined_df["sqi"] >= sqi_thresh).sum()
        log.info("  Clean windows (SQI): %d / %d (%.1f%%)",
                 n_clean, len(combined_df), n_clean / len(combined_df) * 100)
        log.info("  Glucose range:       %.0f – %.0f mg/dL",
                 combined_df["glucose"].min(), combined_df["glucose"].max())
        log.info("  Glucose std:         %.2f mg/dL  (within-patient variability!)",
                 combined_df["glucose"].std())
        log.info("  Archive comparison:  archive within-patient std ≈ 0 mg/dL")
        log.info("  ➜  BigIdeas enables true within-patient glucose tracking evaluation")
    log.info("=" * 60)
    log.info("  Output files in: %s", out_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
