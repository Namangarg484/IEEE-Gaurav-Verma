"""
draw_architecture.py
====================
Render a detailed, publication-quality block diagram of the Multimodal
1D+2D+3D Fusion Network using pure Matplotlib (no external diagram tools).

Output: architecture_diagram.{png,pdf}

Usage:
    python draw_architecture.py
    python draw_architecture.py --out my_diagram.png
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helper primitives
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc="#2c3e50", ec="#1a252f", fs=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs,
        color="white", fontweight="bold", zorder=4,
        multialignment="center", linespacing=1.4,
    )


def _arrow(ax, x1, y1, x2, y2, label="", color="#555"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color,
                        mutation_scale=12),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=7.5, color=color)


def _bracket(ax, x1, x2, y, label):
    """Draw a horizontal bracket to group branches."""
    ax.annotate(
        "", xy=(x1, y), xytext=(x2, y),
        arrowprops=dict(arrowstyle="<->", lw=1.4, color="#888"),
        zorder=1,
    )
    ax.text((x1 + x2) / 2, y + 0.1, label,
            ha="center", va="bottom", fontsize=8, color="#555",
            style="italic")


# ---------------------------------------------------------------------------
def draw(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#f4f6f9")
    ax.set_facecolor("#f4f6f9")

    # ------------------------------------------------------------------ #
    # Column layout (x-positions)                                         #
    # ------------------------------------------------------------------ #
    # C0: raw signal input
    # C1: branch preprocessing
    # C2: CNN encoders
    # C3: projected features
    # C4: cross-attention
    # C5: FC head + output
    C0, C1, C2, C3, C4, C5 = 0.3, 2.3, 5.0, 8.2, 10.3, 13.0

    BW = 1.8   # block width
    BH = 0.85  # block height

    # ------------------------------------------------------------------ #
    # Raw input                                                            #
    # ------------------------------------------------------------------ #
    _box(ax, C0, 4.1, BW, BH, "Raw PPG\nSensor Data\n(ADC values)",
         fc="#2c3e50", ec="#17202a")

    # ------------------------------------------------------------------ #
    # Three branch preprocessing                                          #
    # ------------------------------------------------------------------ #
    _arrow(ax, C0 + BW, 4.55, C1 + 0.0, 7.15, color="#888")   # → 2D branch
    _arrow(ax, C0 + BW, 4.55, C1 + 0.0, 4.55, color="#888")   # → 1D branch
    _arrow(ax, C0 + BW, 4.55, C1 + 0.0, 2.05, color="#888")   # → 3D branch

    _box(ax, C1, 7.0, BW, BH, "CWT Scalogram\n64×64 px\n(mexh wavelet)",
         fc="#16a085", ec="#0e6655")
    _box(ax, C1, 4.1, BW, BH, "Z-Score\nNormalisation\n(1D, 500 pts)",
         fc="#7f8c8d", ec="#5d6d7e")
    _box(ax, C1, 1.2, BW, BH, "Takens' Delay\nτ=5  N=256 pts\n(3-D coord)",
         fc="#e67e22", ec="#ca6f1e")

    # ------------------------------------------------------------------ #
    # CNN encoder blocks                                                   #
    # ------------------------------------------------------------------ #
    _arrow(ax, C1 + BW, 7.42, C2, 7.42, label="(1,64,64)", color="#16a085")
    _arrow(ax, C1 + BW, 4.55, C2, 4.55, label="(1,500)",   color="#7f8c8d")
    _arrow(ax, C1 + BW, 1.62, C2, 1.62, label="(3,256)",   color="#e67e22")

    _box(ax, C2, 6.8, BW + 0.4, BH * 1.35,
         "2D-CNN Branch\nConv2d×3 + BN\nAdaptiveAvgPool\nLinear → 256",
         fc="#2980b9", ec="#1f618d")
    _box(ax, C2, 4.1, BW + 0.4, BH * 1.35,
         "1D-CNN Branch\nConv1d×3 + BN\nAdaptiveAvgPool\nLinear → 256",
         fc="#8e44ad", ec="#6c3483")
    _box(ax, C2, 0.85, BW + 0.4, BH * 1.35,
         "PointNet Branch\nConv1d MLP×3 + BN\nGlobal MaxPool\nLinear → 256",
         fc="#c0392b", ec="#922b21")

    # ------------------------------------------------------------------ #
    # Projected feature labels                                             #
    # ------------------------------------------------------------------ #
    _arrow(ax, C2 + BW + 0.4, 7.68, C3, 7.68, label="f₂ᴅ ∈ ℝ²⁵⁶", color="#2980b9")
    _arrow(ax, C2 + BW + 0.4, 4.85, C3, 4.85, label="f₁ᴅ ∈ ℝ²⁵⁶", color="#8e44ad")
    _arrow(ax, C2 + BW + 0.4, 1.52, C3, 1.52, label="f₃ᴅ ∈ ℝ²⁵⁶", color="#c0392b")

    # Stacking marker
    _box(ax, C3, 5.6, 0.9, 1.55, "[f₁ᴅ\nf₂ᴅ\nf₃ᴅ]",
         fc="#4a4a4a", ec="#2d2d2d")
    _arrow(ax, C3, 7.68, C3 + 0.45, 7.15, color="#2980b9")
    _arrow(ax, C3, 4.85, C3 + 0.45, 6.15, color="#8e44ad")
    _arrow(ax, C3, 1.52, C3 + 0.45, 5.60, color="#c0392b")

    # ------------------------------------------------------------------ #
    # Cross-Attention Fusion                                               #
    # ------------------------------------------------------------------ #
    _arrow(ax, C3 + 0.9, 6.38, C4, 6.38, label="stack(3×256)", color="#4a4a4a")
    _box(ax, C4, 4.8, BW + 0.5, 3.1,
         "Cross-Attention\nFusion Module\n\nlearnable query q\nscaled dot-product\nW_Q, W_K, W_V (128-d)\nsoftmax scores\n→ concat 768-d",
         fc="#f39c12", ec="#d68910", fs=8.5)

    # ------------------------------------------------------------------ #
    # FC Head                                                              #
    # ------------------------------------------------------------------ #
    _arrow(ax, C4 + BW + 0.5, 6.35, C5, 6.35,
           label="768-d fused", color="#f39c12")
    _box(ax, C5, 5.2, BW + 0.2, 2.3,
         "FC Head\n\n768 → 256 (BN+ReLU)\n256 → 64  (BN+ReLU)\nDropout (p=0.3)\n64  → 1",
         fc="#27ae60", ec="#1e8449", fs=8.5)

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    _arrow(ax, C5 + BW + 0.2, 6.35, C5 + BW + 0.85, 6.35, color="#27ae60")
    _box(ax, C5 + BW + 0.85, 5.9, 1.3, 0.9,
         "ŷ\n(mg/dL)",
         fc="#2c3e50", ec="#17202a", fs=10)

    # ------------------------------------------------------------------ #
    # Training details box                                                 #
    # ------------------------------------------------------------------ #
    training_text = (
        "Training config\n"
        "──────────────\n"
        "Loss:  MSE\n"
        "Optim: Adam  lr=0.001\n"
        "Sched: ReduceLROnPlateau\n"
        "Clip:  grad_norm ≤ 1.0\n"
        "Dropout: 0.30\n"
        "Kaiming init"
    )
    ax.text(
        0.25, 0.2, training_text,
        transform=ax.transAxes, va="bottom", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#aaa"),
    )

    # ------------------------------------------------------------------ #
    # Title                                                                #
    # ------------------------------------------------------------------ #
    ax.set_title(
        "Multimodal 1D + 2D + 3D Deep Fusion Network Architecture\n"
        "for Non-Invasive Blood Glucose Prediction from PPG Signals",
        fontsize=14, fontweight="bold", pad=12,
    )

    plt.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fp = out_path.with_suffix(f".{ext}")
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", fp)

    plt.close(fig)


# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Draw network architecture diagram",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "architecture_diagram.png"),
        metavar="PATH",
        help="Output file (extension overridden; both .png and .pdf are saved)",
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
    draw(Path(args.out))
    logger.info("Architecture diagram complete.")
