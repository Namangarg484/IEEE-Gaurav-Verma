"""
generate_extra_figures.py
=========================
Generate supplementary IEEE paper figures from REAL model inference (no fabrication).

Figures produced
----------------
    Paper_Figure_4_Loss_Curve.png     — train/val MSE from training_log.csv
    Paper_Figure_5_SNR.png            — SNR robustness: RMSE vs noise level
    Paper_Figure_6_Attention.png      — attention weight distribution per branch

All data is derived from actual model runs — no hardcoded or random arrays.

Usage:
    python generate_extra_figures.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fusion_data_pipeline import PPGFusionDataset, build_dataloader, build_train_val_datasets
from fusion_network_model import MultimodalFusionNet
from train_fusion_model import get_device

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent
CSV_PATH  = ROOT / "Archive_Dataset" / "clean-dataset.csv"
MODEL_PTH = ROOT / "fusion_model_v2_mac.pth"

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14})


# ---------------------------------------------------------------------------
# Helper — load model once
# ---------------------------------------------------------------------------
def _load_model(device: torch.device) -> nn.Module:
    if not MODEL_PTH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PTH}\n"
            "Run train_fusion_model.py first."
        )
    model = MultimodalFusionNet().to(device)
    model.load_state_dict(torch.load(str(MODEL_PTH), weights_only=True))
    model.eval()
    logger.info("Loaded model: %s", MODEL_PTH)
    return model


# ---------------------------------------------------------------------------
# Figure 4 — Training Convergence (Loss Curve)
# ---------------------------------------------------------------------------
def gen_figure_4() -> None:
    """Plot real train/val MSE from training_log.csv."""
    logger.info("Generating Figure 4: Training loss curve …")
    log_path = ROOT / "training_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"training_log.csv not found at {log_path}")

    log        = pd.read_csv(log_path)
    epochs     = log["epoch"].values
    train_loss = log["train_mse"].values
    val_loss   = log["val_mse"].values
    best_ep    = int(log.loc[log["val_mse"].idxmin(), "epoch"])

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, "o-",  color="#1f77b4", lw=2.5, ms=6, label="Train MSE")
    plt.plot(epochs, val_loss,   "s--", color="#ff7f0e", lw=2.5, ms=6, label="Validation MSE")
    plt.axvline(x=best_ep, color="red", ls="--", lw=2,
                label=f"Best epoch = {best_ep}  (Val MSE = {val_loss[best_ep-1]:.2f})")

    plt.title("Fig. 4: Training Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("MSE  (mg/dL²)")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    out = ROOT / "Paper_Figure_4_Loss_Curve.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("  Saved: %s", out)


# ---------------------------------------------------------------------------
# Figure 5 — SNR Robustness (real inference with synthetic noise)
# ---------------------------------------------------------------------------
def gen_figure_5() -> None:
    """Add Gaussian noise at various SNR levels; measure actual model RMSE."""
    logger.info("Generating Figure 5: SNR robustness experiment …")

    device = get_device()
    model  = _load_model(device)

    df         = pd.read_csv(CSV_PATH)
    patients   = df["Patient_Id"].unique()
    val_df     = df[df["Patient_Id"].isin(patients[18:])].reset_index(drop=True)

    criterion  = nn.MSELoss(reduction="sum")
    snr_levels = [20, 15, 10, 5, 0, -5]

    rmse_fusion = []
    rmse_1d_only = []

    for snr_db in snr_levels:
        val_ds     = PPGFusionDataset(dataframe=val_df.copy(), window_size=500)
        val_loader = build_dataloader(val_ds, batch_size=32, shuffle=False)

        total_loss = 0.0
        n_total    = 0
        total_loss_1d = 0.0

        with torch.no_grad():
            for batch in val_loader:
                s1  = batch["1d_signal"].to(device)
                s2  = batch["2d_spectrogram"].to(device)
                s3  = batch["3d_phase"].to(device)
                lbl = batch["label"].to(device)
                n   = s1.size(0)

                # Add Gaussian noise to 1D signal at specified SNR
                sig_power   = s1.pow(2).mean()
                noise_power = sig_power / (10 ** (snr_db / 10.0))
                noise       = torch.randn_like(s1) * noise_power.sqrt()
                s1_noisy    = s1 + noise

                # Fusion model (all branches with noisy 1D)
                out_fusion = model(s1_noisy, s2, s3)
                total_loss += criterion(out_fusion, lbl).item()

                # 1D-only baseline: zero out 2D and 3D contributions
                zero_2d = torch.zeros_like(s2)
                zero_3d = torch.zeros_like(s3)
                out_1d  = model(s1_noisy, zero_2d, zero_3d)
                total_loss_1d += criterion(out_1d, lbl).item()

                n_total += n

        rmse_fusion.append(float(np.sqrt(total_loss / n_total)))
        rmse_1d_only.append(float(np.sqrt(total_loss_1d / n_total)))
        logger.info("  SNR %+3d dB  Fusion RMSE = %.2f  1D-only RMSE = %.2f",
                    snr_db, rmse_fusion[-1], rmse_1d_only[-1])

    plt.figure(figsize=(7, 5))
    plt.plot(snr_levels, rmse_fusion,  "o-", color="#1f77b4", lw=2.5, ms=7,
             label="Fusion (1D+2D+3D)")
    plt.plot(snr_levels, rmse_1d_only, "s--", color="#d62728", lw=2.5, ms=7,
             label="1D Branch Only")
    plt.gca().invert_xaxis()
    plt.title("Fig. 5: SNR Robustness — Fusion vs 1D-Only")
    plt.xlabel("SNR (dB)  [← increasing noise]")
    plt.ylabel("RMSE  (mg/dL)")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    out = ROOT / "Paper_Figure_5_SNR.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("  Saved: %s", out)


# ---------------------------------------------------------------------------
# Figure 6 — Attention Weight Distribution
# ---------------------------------------------------------------------------
def gen_figure_6() -> None:
    """Plot per-branch attention weights extracted from real val-set inference."""
    logger.info("Generating Figure 6: Attention weight distribution …")

    device = get_device()
    model  = _load_model(device)

    df       = pd.read_csv(CSV_PATH)
    patients = df["Patient_Id"].unique()
    val_df   = df[df["Patient_Id"].isin(patients[18:])].reset_index(drop=True)
    val_ds   = PPGFusionDataset(dataframe=val_df, window_size=500)
    loader   = build_dataloader(val_ds, batch_size=32, shuffle=False)

    all_weights = []
    with torch.no_grad():
        for batch in loader:
            s1 = batch["1d_signal"].to(device)
            s2 = batch["2d_spectrogram"].to(device)
            s3 = batch["3d_phase"].to(device)
            _, w = model.forward_with_attention(s1, s2, s3)
            all_weights.append(w.cpu().numpy())

    weights_np = np.vstack(all_weights)   # (N_samples, 3)
    means      = weights_np.mean(axis=0)  # [w1d, w2d, w3d]
    stds       = weights_np.std(axis=0)

    labels = ["1D Branch\n(Temporal)", "2D Branch\n(CWT)", "3D Branch\n(Phase)"]
    colors = ["#8e44ad", "#2980b9", "#c0392b"]
    x      = np.arange(3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Bar chart of mean attention weights
    bars = ax1.bar(x, means, color=colors, edgecolor="black", linewidth=1.2,
                   capsize=6, width=0.5)
    ax1.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=6, lw=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean Attention Weight")
    ax1.set_title("Mean ± Std Attention Weights\n(Validation Set)")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.4)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    # Violin / box plot of weight distributions
    vp = ax2.violinplot(
        [weights_np[:, i] for i in range(3)],
        positions=x, showmeans=True, showmedians=True,
    )
    for pc, color in zip(vp["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Attention Weight")
    ax2.set_title("Attention Weight Distribution\n(Validation Set)")
    ax2.grid(axis="y", alpha=0.4)

    logger.info("  Branch weights (mean ± std):")
    for i, lbl in enumerate(["1D", "2D", "3D"]):
        logger.info("    %s: %.4f ± %.4f", lbl, means[i], stds[i])

    plt.suptitle("Fig. 6: Cross-Attention Branch Contribution Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = ROOT / "Paper_Figure_6_Attention.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("  Saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not CSV_PATH.exists():
        logger.error("CSV not found: %s", CSV_PATH)
        sys.exit(1)

    errors = []
    for name, fn in [("Figure 4", gen_figure_4),
                     ("Figure 5", gen_figure_5),
                     ("Figure 6", gen_figure_6)]:
        try:
            fn()
        except Exception as exc:
            logger.error("%s FAILED: %s", name, exc, exc_info=True)
            errors.append(name)

    if errors:
        logger.error("Failed figures: %s", errors)
        sys.exit(1)

    logger.info("\nAll extra figures generated successfully.")
