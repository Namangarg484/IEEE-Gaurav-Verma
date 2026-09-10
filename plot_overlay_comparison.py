"""
plot_overlay_comparison.py
==========================
Overlay predicted vs reference blood-glucose values for every validation
window, colour-coded by patient.  Also plots a residual histogram.

Outputs
-------
    overlay_comparison.png   — reference vs predicted scatter + residual histogram

Usage:
    python plot_overlay_comparison.py
    python plot_overlay_comparison.py --csv /path/to/clean-dataset.csv \
                                       --model fusion_model_v2_mac.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from fusion_data_pipeline import PPGFusionDataset, build_dataloader
from fusion_network_model import MultimodalFusionNet
from train_fusion_model import get_device

logger = logging.getLogger(__name__)
ROOT   = Path(__file__).parent

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: nn.Module,
    csv_path: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (references, predictions, patient_ids) arrays for the validation set."""
    import pandas as pd

    df       = pd.read_csv(csv_path)
    patients = df["Patient_Id"].unique()
    val_df   = df[df["Patient_Id"].isin(patients[18:])].reset_index(drop=True)

    val_ds     = PPGFusionDataset(dataframe=val_df, window_size=500)
    val_loader = build_dataloader(val_ds, batch_size=32, shuffle=False)

    # Build patient_id array from dataframe (one entry per 500-row window)
    n_wins  = len(val_ds)
    win_pids = np.array([
        int(val_df.iloc[i * 500]["Patient_Id"]) for i in range(n_wins)
    ])

    all_refs  = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            s1  = batch["1d_signal"].to(device)
            s2  = batch["2d_spectrogram"].to(device)
            s3  = batch["3d_phase"].to(device)
            lbl = batch["label"]

            out = model(s1, s2, s3).cpu().squeeze(-1)
            all_refs.append(lbl.numpy())
            all_preds.append(out.numpy())

    return (
        np.concatenate(all_refs).flatten(),
        np.concatenate(all_preds).flatten(),
        win_pids,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    refs: np.ndarray,
    preds: np.ndarray,
    patient_ids: np.ndarray,
    save_path: Path,
) -> None:
    unique_pids = np.unique(patient_ids)
    cmap        = plt.colormaps["tab10"].resampled(len(unique_pids))
    residuals   = preds - refs

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Predicted vs Reference Glucose — Validation Set", fontsize=14)

    # --- Panel 1: Reference vs Predicted scatter ----------------------------
    ax = axes[0]
    for i, pid in enumerate(unique_pids):
        mask = patient_ids == pid
        ax.scatter(
            refs[mask], preds[mask],
            color=cmap(i), s=22, alpha=0.75, edgecolors="none",
            label=f"Patient {int(pid)}",
        )

    lo = min(refs.min(), preds.min()) - 5
    hi = max(refs.max(), preds.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="Ideal (y = x)")
    ax.fill_between(
        [lo, hi],
        [lo * 1.1 - 11, hi * 1.1 - 11],
        [lo * 0.9 + 9, hi * 0.9 + 9],
        alpha=0.07, color="gray", label="±10% / ±10 mg/dL band",
    )

    mae  = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2   = float(1 - np.sum(residuals ** 2) / np.sum((refs - refs.mean()) ** 2))
    ax.text(
        0.04, 0.96,
        f"MAE  = {mae:.2f} mg/dL\nRMSE = {rmse:.2f} mg/dL\nR²   = {r2:.4f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Reference Glucose (mg/dL)")
    ax.set_ylabel("Predicted Glucose (mg/dL)")
    ax.set_title("Reference vs Predicted")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.35)

    # --- Panel 2: Residual histogram ----------------------------------------
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color="#2980b9", edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax2.axvline(0, color="red", lw=2, linestyle="--", label="Zero bias")
    ax2.axvline(mae,  color="orange", lw=1.5, linestyle=":",
                label=f"+MAE ({mae:.1f})")
    ax2.axvline(-mae, color="orange", lw=1.5, linestyle=":")
    ax2.set_xlabel("Residual  (Predicted − Reference)  [mg/dL]")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.35)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Plot predicted vs reference glucose overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",   default=str(root / "Archive_Dataset" / "clean-dataset.csv"))
    p.add_argument("--model", default=str(root / "fusion_model_v2_mac.pth"))
    p.add_argument("--out",   default=str(root / "overlay_comparison.png"))
    return p.parse_args()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = _parse()

    for fp, label in [(args.csv, "CSV"), (args.model, "Model")]:
        if not Path(fp).exists():
            logger.error("%s not found: %s", label, fp)
            sys.exit(1)

    device = get_device()
    model  = MultimodalFusionNet().to(device)
    model.load_state_dict(
        torch.load(args.model, weights_only=True)
    )
    logger.info("Loaded model from %s", args.model)

    logger.info("Running inference on validation set …")
    refs, preds, pids = run_inference(model, args.csv, device)
    logger.info("  Inference complete: %d windows", len(refs))

    plot_comparison(refs, preds, pids, Path(args.out))
    logger.info("Overlay comparison saved.")
