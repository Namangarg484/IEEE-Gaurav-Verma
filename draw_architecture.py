"""
draw_architecture.py
====================
Block diagram of the 1D-2D-3D fusion network for the paper (pure Matplotlib).

    python draw_architecture.py            # -> figures/architecture_diagram.{pdf,png}

The labels mirror fusion_network_model.py and ppg_data.py exactly: 320-sample
windows at 64 Hz, 64x64 scalogram, 256 Takens points, three 256-d encoders,
branch-specific key projections W_i, one learnable query q, softmax weights
that scale the original embeddings, 768-d concatenation and the FC head.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures", "architecture_diagram")
INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FILL = "#f4f3f0"
TITLE_FS, BODY_FS, ARROW_FS = 7.0, 5.8, 5.8

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def box(ax, x, y, w, h, title, body, color=EDGE):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                         facecolor=FILL, edgecolor=color, linewidth=1.0, zorder=3))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=TITLE_FS, color=INK,
            fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + 0.1, body, ha="center", va="bottom", fontsize=BODY_FS, color=MUTED,
            zorder=4, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label="", color=EDGE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": color, "mutation_scale": 8,
                            "shrinkA": 0, "shrinkB": 0})
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, label, ha="center", va="bottom", fontsize=ARROW_FS,
                color=color)


def draw() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 15.5); ax.set_ylim(0.6, 6.6); ax.axis("off")
    rows = {"2d": 5.0, "1d": 3.0, "3d": 1.0}
    H = 1.35
    mid = {k: v + H / 2 for k, v in rows.items()}

    # input --------------------------------------------------------------
    box(ax, 0.15, rows["1d"], 1.95, H, "PPG window", "0.5–8 Hz band-pass\n64 Hz, 5 s = 320 samples\nz-scored")
    for k, c in (("2d", AQUA), ("1d", BLUE), ("3d", ORANGE)):
        arrow(ax, 2.1, mid["1d"], 2.55, mid[k], color=c)

    # views --------------------------------------------------------------
    VX, VW = 2.55, 2.45
    box(ax, VX, rows["2d"], VW, H, "CWT scalogram", "Mexican-hat, 64 scales\nresized to 64 × 64\n1 × 64 × 64", AQUA)
    box(ax, VX, rows["1d"], VW, H, "Waveform", "raw window\n1 × 320", BLUE)
    box(ax, VX, rows["3d"], VW, H, "Takens embedding", "[x(i), x(i+5), x(i+10)]\n256 points\n3 × 256", ORANGE)

    # encoders -----------------------------------------------------------
    EX, EW = 5.6, 2.7
    for k, c in (("2d", AQUA), ("1d", BLUE), ("3d", ORANGE)):
        arrow(ax, VX + VW, mid[k], EX, mid[k], color=c)
    box(ax, EX, rows["2d"], EW, H, "2-D CNN", "Conv2d ×3 (3×3), BN, ReLU\nmax-pool, GAP\nLinear 128→256", AQUA)
    box(ax, EX, rows["1d"], EW, H, "1-D CNN", "Conv1d ×3 (k 7/5/3, s 2)\nBN, ReLU, GAP\nLinear 128→256", BLUE)
    box(ax, EX, rows["3d"], EW, H, "PointNet", "shared MLP 3→64→128→256\nBN, ReLU\nglobal max-pool", ORANGE)

    # attention ----------------------------------------------------------
    AX, AW = 9.1, 3.0
    for k, c, name in (("2d", AQUA, "f₂"), ("1d", BLUE, "f₁"), ("3d", ORANGE, "f₃")):
        arrow(ax, EX + EW, mid[k], AX, mid[k], name, c)
    box(ax, AX, rows["3d"], AW, rows["2d"] + H - rows["3d"], "Learnable-query attention",
        "kᵢ = Wᵢ fᵢ   (256→128)\n"
        "eᵢ = qᵀkᵢ / √128\n"
        "α = softmax(e₁, e₂, e₃)\n"
        "z = [α₁f₁; α₂f₂; α₃f₃]\n"
        "768-d", YELLOW)

    # head + output ------------------------------------------------------
    HX, HW = 12.75, 2.1
    arrow(ax, AX + AW, mid["1d"], HX, mid["1d"], "z")
    box(ax, HX, rows["1d"] - 0.35, HW, H + 0.7, "FC head",
        "768→256, BN, ReLU\nDropout 0.3\n256→64, ReLU\n64→1")
    arrow(ax, HX + HW, mid["1d"], HX + HW + 0.35, mid["1d"])
    ax.text(HX + HW + 0.4, mid["1d"], "ŷ", ha="left", va="center", fontsize=8, color=INK)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    draw()
