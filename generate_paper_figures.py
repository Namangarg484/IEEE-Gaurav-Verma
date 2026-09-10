"""
generate_paper_figures.py
=========================
Generate publication-quality IEEE paper figures from real data.

Figures produced
----------------
    Paper_Figure_1_Multimodal_Pipeline.{png,pdf}  — 1D/2D/3D transformation panel
    Paper_Figure_2_Raw_Variance.{png,pdf}          — raw PPG overlays (5 glucose levels)
    Paper_Figure_3_Architecture.{png,pdf}          — network architecture diagram

Usage:
    python generate_paper_figures.py
    python generate_paper_figures.py --csv /path/to/clean-dataset.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       10,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})


# ---------------------------------------------------------------------------
# Figure 1 — Multimodal Transformation Pipeline
# ---------------------------------------------------------------------------
def gen_figure_1(csv_path: str, out_dir: Path) -> None:
    """3-panel figure: raw 1D signal / 2D CWT scalogram / 3D phase attractor."""
    logger.info("Generating Figure 1: Multimodal Pipeline …")
    from fusion_data_pipeline import PPGFusionDataset

    ds     = PPGFusionDataset(csv_file=csv_path, window_size=500)
    sample = ds[min(350, len(ds) - 1)]

    sig1d   = sample["1d_signal"].numpy().flatten()
    spec2d  = sample["2d_spectrogram"].numpy()[0]
    phase3d = sample["3d_phase"].numpy()
    glucose = float(sample["label"].item())

    fig = plt.figure(figsize=(15, 5))

    # Panel A — 1D temporal signal
    ax1 = fig.add_subplot(131)
    ax1.plot(sig1d, color="#1f77b4", linewidth=1.5)
    ax1.set_title(f"(A) 1D Temporal PPG Signal\nGlucose: {glucose:.1f} mg/dL")
    ax1.set_xlabel("Time Samples")
    ax1.set_ylabel("Normalised Amplitude")
    ax1.grid(True, alpha=0.3)

    # Panel B — 2D CWT scalogram
    ax2 = fig.add_subplot(132)
    im  = ax2.imshow(spec2d, aspect="auto", cmap="magma", origin="lower")
    ax2.set_title("(B) 2D CWT Time-Frequency Scalogram")
    ax2.set_xlabel("Temporal Features")
    ax2.set_ylabel("Frequency / Scale Bands")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Panel C — 3D Takens attractor
    ax3 = fig.add_subplot(133, projection="3d")
    sc  = ax3.scatter(
        phase3d[0], phase3d[1], phase3d[2],
        c=phase3d[2], cmap="viridis", s=6, alpha=0.8,
    )
    ax3.plot(phase3d[0], phase3d[1], phase3d[2], color="gray", alpha=0.3, linewidth=0.5)
    ax3.set_title("(C) 3D Takens' Phase-Space Attractor")
    ax3.set_xlabel("x(t)")
    ax3.set_ylabel("x(t+τ)")
    ax3.set_zlabel("x(t+2τ)")
    ax3.view_init(elev=25, azim=45)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fp = out_dir / f"Paper_Figure_1_Multimodal_Pipeline.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        logger.info("  Saved: %s", fp)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Raw PPG Overlays (physiological variance)
# ---------------------------------------------------------------------------
def gen_figure_2(csv_path: str, out_dir: Path) -> None:
    """Overlay of 5 raw PPG windows at diverse glucose levels."""
    logger.info("Generating Figure 2: Raw Physiological Variance …")
    df      = pd.read_csv(csv_path)
    indices = [0, 280, 350, 625, 695]   # hand-picked diverse glucose windows

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.Set2(np.linspace(0, 1, len(indices)))

    for i, idx in enumerate(indices):
        start = idx * 500
        end   = start + 500
        if end > len(df):
            logger.warning("Index %d exceeds dataset length; skipping.", idx)
            continue
        chunk  = df.iloc[start:end]
        raw    = chunk["PPG_Signal"].values.astype(np.float32)
        glc    = float(np.mean(chunk["Glucose_level"].values))
        ax.plot(raw, label=f"Glucose: {glc:.1f} mg/dL",
                color=colors[i], alpha=0.85, linewidth=1.5)

    ax.axvspan(200, 300, color="gray", alpha=0.12, label="Dicrotic Notch Zone")
    ax.set_title("Raw PPG ADC Overlays (before Z-score Normalisation)", fontsize=14)
    ax.set_xlabel("Time Samples")
    ax.set_ylabel("Raw Sensor Amplitude (ADC)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fp = out_dir / f"Paper_Figure_2_Raw_Variance.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        logger.info("  Saved: %s", fp)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Architecture Diagram (pure matplotlib)
# ---------------------------------------------------------------------------
def gen_figure_3(out_dir: Path) -> None:
    """Clean architecture block diagram."""
    logger.info("Generating Figure 3: Architecture Diagram …")
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#f9f9f9")

    def box(x, y, w, h, text, fc, ec, fs=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=3,
                multialignment="center")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#444"))

    # Input
    box(0.3, 3.0, 1.8, 1.0, "Raw 1D PPG\nSignal\n(1×500)", "#2c3e50", "#1a252f", fs=8)

    # Transforms
    arrow(2.1, 4.3, 3.1, 5.5)
    arrow(2.1, 3.5, 3.1, 3.5)
    arrow(2.1, 3.0, 3.1, 1.5)

    box(3.1, 5.0, 2.2, 1.0, "CWT\nSpectrogram", "#16a085", "#0e6655", fs=8)
    box(3.1, 3.1, 2.2, 0.9, "Z-Score\nNorm", "#7f8c8d", "#5d6d7e", fs=8)
    box(3.1, 1.0, 2.2, 1.0, "Takens'\nDelay τ", "#e67e22", "#ca6f1e", fs=8)

    # CNN branches
    arrow(5.3, 5.5, 6.2, 5.5)
    arrow(5.3, 3.5, 6.2, 3.5)
    arrow(5.3, 1.5, 6.2, 1.5)

    box(6.2, 5.0, 2.2, 1.0, "2D-CNN\n→ R^256", "#2980b9", "#1f618d", fs=8)
    box(6.2, 3.1, 2.2, 0.9, "1D-CNN\n→ R^256", "#8e44ad", "#6c3483", fs=8)
    box(6.2, 1.0, 2.2, 1.0, "PointNet\n→ R^256", "#c0392b", "#922b21", fs=8)

    # Cross-Attention
    arrow(8.4, 5.5, 9.2, 4.5)
    arrow(8.4, 3.5, 9.2, 3.8)
    arrow(8.4, 1.5, 9.2, 3.1)

    box(9.2, 2.8, 2.4, 2.0,
        "Cross-Attention\nFusion\n(learnable q)\n→ R^768",
        "#f39c12", "#d68910", fs=8)

    # FC head
    arrow(11.6, 3.8, 12.3, 3.8)
    box(12.3, 3.2, 1.5, 1.2,
        "FC Head\n768→256→1\nDropout 0.3",
        "#27ae60", "#1e8449", fs=8)

    # Output
    arrow(13.8, 3.8, 14.2, 3.8)
    ax.text(13.85, 3.95, "Glucose\n(mg/dL)", fontsize=8, ha="left", color="#2c3e50")

    ax.set_title(
        "Multimodal 1D+2D+3D Fusion Network Architecture",
        fontsize=14, fontweight="bold", pad=10,
    )

    for ext in ("png", "pdf"):
        fp = out_dir / f"Paper_Figure_3_Architecture.{ext}"
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        logger.info("  Saved: %s", fp)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Generate IEEE paper figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=str(root / "Archive_Dataset" / "clean-dataset.csv"),
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args    = _parse()
    out_dir = Path(__file__).parent

    if not Path(args.csv).exists():
        logger.error("CSV not found: %s", args.csv)
        sys.exit(1)

    try:
        gen_figure_1(args.csv, out_dir)
        gen_figure_2(args.csv, out_dir)
        gen_figure_3(out_dir)
        logger.info("\nAll paper figures generated successfully.")
    except Exception as exc:
        logger.error("Figure generation failed: %s", exc, exc_info=True)
        sys.exit(1)
