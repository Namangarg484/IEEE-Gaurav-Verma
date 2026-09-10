"""
generate_generalization_figure.py
==================================
Generates the cross-dataset generalisation figure for the IEEE paper.
All numbers are hard-coded from real experiment logs (linear_probe.py terminal
output and prepare_generalization_data.py output).

Panel A : MAE comparison bar chart (in-distribution / zero-shot / fine-tuned)
Panel B : Per-epoch val MAE convergence curve during fine-tuning

Output  : Paper_CrossDataset_Generalization.png  (150 dpi, ≈ 1200×525 px)
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_generalization")
os.makedirs("/tmp/matplotlib_generalization", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Real numbers from experiment logs ─────────────────────────────────────────
INDIST_MAE    = 6.89     # Archive val-set, MAE (training_log + ablation_results.json)
INDIST_RMSE   = 9.08

ZEROSHOT_MAE  = 15.20    # linear_probe.py terminal: "Zero-shot MAE"
ZEROSHOT_RMSE = 18.29

FINETUNE_MAE  = 13.99    # linear_probe.py terminal: "Fine-tuned MAE (after 15 epochs)"
FINETUNE_RMSE = 17.94

# Per-epoch val MAE during fine-tuning — from linear_probe.py terminal log
EPOCH_VAL_MAE = [
    32.51, 20.80, 14.44, 17.16, 14.97,
    15.72, 22.69, 23.14, 14.61, 18.77,
    13.25, 14.34, 16.78, 14.55, 13.99,
]
EPOCHS = list(range(1, len(EPOCH_VAL_MAE) + 1))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
fig.suptitle(
    "Cross-Dataset Generalisation  —  PhysioNet BIG IDEAs (Subject 001, $n$=411 test)",
    fontsize=8.5, fontweight="bold", y=1.01,
)

# ─── Panel A: MAE bar chart ───────────────────────────────────────────────────
labels  = [
    "In-distribution\n(Archive, $n$=280)",
    "Zero-shot\n(PhysioNet, frozen)",
    "Fine-tuned\n(attn+head, 15 ep)",
]
maes    = [INDIST_MAE, ZEROSHOT_MAE, FINETUNE_MAE]
colors  = ["#2196F3", "#FF9800", "#4CAF50"]

bars = ax1.bar(labels, maes, color=colors, width=0.52,
               edgecolor="k", linewidth=0.6, zorder=3)
for bar, val in zip(bars, maes):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.35,
        f"{val:.2f}",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
    )

ax1.set_ylabel("MAE (mg/dL)", fontsize=9)
ax1.set_title("(A) MAE Comparison", fontsize=9, pad=6)
ax1.set_ylim(0, 19)
ax1.tick_params(axis="x", labelsize=7.5)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Annotate improvement arrow
ax1.annotate(
    "",
    xy=(2, FINETUNE_MAE + 0.3),
    xytext=(1, ZEROSHOT_MAE - 0.3),
    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0),
)
ax1.text(
    1.6, (FINETUNE_MAE + ZEROSHOT_MAE) / 2,
    r"$\Delta$MAE$=$−1.21",
    fontsize=7, color="#555555", ha="center", va="center",
    rotation=75,
)

# ─── Panel B: Fine-tuning convergence curve ───────────────────────────────────
ax2.plot(
    EPOCHS, EPOCH_VAL_MAE, "o-",
    color="#4CAF50", linewidth=1.6, markersize=4, label="Val MAE (fine-tuned)",
    zorder=3,
)
ax2.axhline(
    y=ZEROSHOT_MAE, color="#FF9800", linewidth=1.3,
    linestyle="--", label=f"Zero-shot ({ZEROSHOT_MAE:.2f} mg/dL)", zorder=2,
)
ax2.axhline(
    y=INDIST_MAE, color="#2196F3", linewidth=1.0,
    linestyle=":", label=f"In-dist. ({INDIST_MAE:.2f} mg/dL)", zorder=2,
)

ax2.set_xlabel("Fine-tuning Epoch", fontsize=9)
ax2.set_ylabel("Val MAE (mg/dL)", fontsize=9)
ax2.set_title("(B) Fine-tuning Convergence", fontsize=9, pad=6)
ax2.legend(fontsize=7, loc="upper right")
ax2.set_xlim(0.5, 15.5)
ax2.set_ylim(0, 37)
ax2.grid(alpha=0.3, zorder=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Annotate best point
best_ep  = int(np.argmin(EPOCH_VAL_MAE)) + 1
best_mae = min(EPOCH_VAL_MAE)
ax2.annotate(
    f"Best: {best_mae:.2f}\n(ep {best_ep})",
    xy=(best_ep, best_mae),
    xytext=(best_ep - 4, best_mae + 9),
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    fontsize=7, color="#4CAF50",
)

plt.tight_layout()
out_path = "Paper_CrossDataset_Generalization.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
