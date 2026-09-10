"""
visualize_multimodal_data.py
============================
Generate per-sample multi-panel visualisations of the three modalities
(1D temporal / 2D CWT spectrogram / 3D Takens phase-space attractor).

Saves one PNG per sample to a Visualizations/ sub-folder.

Usage:
    python visualize_multimodal_data.py
    python visualize_multimodal_data.py --csv /path/to/clean-dataset.csv \
                                        --indices 0 100 350 625 695 \
                                        --out Visualizations
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fusion_data_pipeline import PPGFusionDataset

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "figure.dpi":      120,
})


# ---------------------------------------------------------------------------
def visualize_sample(ds: PPGFusionDataset, win_idx: int, out_dir: Path) -> None:
    """Render a 3-panel figure for a single window index and save to out_dir."""
    sample = ds[win_idx]

    sig1d   = sample["1d_signal"].numpy().flatten()             # (500,)
    spec2d  = sample["2d_spectrogram"].numpy()[0]               # (64, 64)
    phase3d = sample["3d_phase"].numpy()                        # (3, 256)
    glucose = float(sample["label"].item())

    fig = plt.figure(figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle(
        f"Sample {win_idx} — Glucose: {glucose:.1f} mg/dL",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel A: 1D signal ------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(sig1d, color="#2980b9", linewidth=1.5)
    ax1.set_title("1D Temporal Signal (Z-score Normalised)")
    ax1.set_xlabel("Time Sample")
    ax1.set_ylabel("Normalised Amplitude")
    ax1.grid(True, alpha=0.3)
    # annotate peak
    peak_idx = int(np.argmax(sig1d))
    ax1.annotate(
        f"Peak\n{sig1d[peak_idx]:.2f}",
        xy=(peak_idx, sig1d[peak_idx]),
        xytext=(peak_idx - 80, sig1d[peak_idx] - 0.5),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=9,
    )

    # ---- Panel B: CWT scalogram --------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(
        spec2d, aspect="auto", cmap="magma", origin="lower",
        extent=[0, spec2d.shape[1], 0, spec2d.shape[0]],
    )
    ax2.set_title("2D CWT Time-Frequency Scalogram")
    ax2.set_xlabel("Time Bins")
    ax2.set_ylabel("Frequency / Scale Band")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Normalised Power")

    # ---- Panel C: 3D phase attractor ----------------------------------------
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    n_pts = phase3d.shape[1]
    color_map = plt.cm.viridis(np.linspace(0, 1, n_pts))
    ax3.scatter(
        phase3d[0], phase3d[1], phase3d[2],
        c=color_map, s=8, alpha=0.85,
    )
    ax3.plot(
        phase3d[0], phase3d[1], phase3d[2],
        color="gray", alpha=0.25, linewidth=0.6,
    )
    ax3.set_title("3D Takens' Phase-Space Attractor")
    ax3.set_xlabel("x(t)")
    ax3.set_ylabel("x(t+τ)")
    ax3.set_zlabel("x(t+2τ)")
    ax3.view_init(elev=25, azim=45)

    out_path = out_dir / f"sample_{win_idx:04d}_glucose_{glucose:.1f}mgdL.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)


# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Visualise multimodal PPG data for selected window indices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=str(root / "Archive_Dataset" / "clean-dataset.csv"),
    )
    p.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0, 280, 350, 625, 695],
        metavar="N",
        help="Window indices in the dataset to visualise",
    )
    p.add_argument(
        "--out",
        default=str(root / "Visualizations"),
        metavar="DIR",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = _parse()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out)

    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    logger.info("Loading dataset …")
    ds = PPGFusionDataset(csv_file=str(csv_path), window_size=500)
    logger.info("  Total windows: %d", len(ds))

    errors = []
    for idx in args.indices:
        if idx >= len(ds):
            logger.warning("Index %d >= dataset size %d — skipping", idx, len(ds))
            continue
        try:
            visualize_sample(ds, idx, out_dir)
        except Exception as exc:
            logger.error("Sample %d failed: %s", idx, exc, exc_info=True)
            errors.append(idx)

    if errors:
        logger.error("Failed indices: %s", errors)
        sys.exit(1)

    logger.info("\nAll visualizations saved to: %s", out_dir)
