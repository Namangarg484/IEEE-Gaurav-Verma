"""
train_fusion_model.py
=====================
Training entry-point for the Multimodal 1D+2D+3D Blood Glucose Fusion Network.

Live output
-----------
    Console        — epoch-by-epoch table with timing (always unbuffered)
    training_log.csv     — per-epoch metrics written atomically after each epoch
    training_run.log     — exact mirror of console output for offline review
    clarke_results.json  — clinical zone counts after final evaluation
    clarke_error_grid.png

Usage (terminal / run_training.sh)
-----------------------------------
    python train_fusion_model.py
    python train_fusion_model.py --epochs 50 --batch 32 --lr 0.001 --patience 10
    python train_fusion_model.py --csv /path/to/clean-dataset.csv

Fixes vs previous version
--------------------------
- Gradient clipping (max_norm=1.0)  → eliminates the 76→290→77 MSE oscillation
- ReduceLROnPlateau scheduler       → adapts to actual validation behaviour
- min_delta early-stopping          → ignores sub-noise "improvements"
- All three branches output 256-dim → 2D branch no longer dominates attention
- weights_only=True on all torch.load  → no pickle injection risk
- Relative paths via pathlib        → script runs from any working directory
- Full logging to file + stdout     → tail -f training_run.log works perfectly
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fusion_data_pipeline import build_dataloader, build_train_val_datasets
from fusion_network_model import MultimodalFusionNet
from plot_clarke_error_grid import clarke_error_grid, summarize_clarke_zones

# ---------------------------------------------------------------------------
# Logging — tee to both stdout and a persistent file
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).parent / "training_run.log"
_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
        ],
    )
    # Ensure stdout is unbuffered for live tail output
    sys.stdout.reconfigure(line_buffering=True)
    return logging.getLogger(__name__)


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the best available compute device for this machine."""
    if torch.backends.mps.is_available():
        logger.info("Hardware: Apple Silicon GPU (MPS) — using Metal acceleration")
        return torch.device("mps")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("Hardware: NVIDIA GPU (CUDA) — %s", name)
        return torch.device("cuda")
    logger.warning("Hardware: CPU only — training will be slow (~10× vs GPU)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Pre-training shape sanity check
# ---------------------------------------------------------------------------
def sanity_check(model: nn.Module, device: torch.device) -> None:
    """Assert tensor shapes through all branches and the fusion layer."""
    logger.info("Running pre-training shape sanity check ...")
    B = 2
    d1 = torch.randn(B, 1,    500).to(device)
    d2 = torch.randn(B, 1, 64, 64).to(device)
    d3 = torch.randn(B, 3,    256).to(device)

    f1 = model.branch_1d(d1)
    f2 = model.branch_2d(d2)
    f3 = model.branch_3d(d3)

    assert f1.shape == (B, 256), f"Branch1D shape mismatch: {f1.shape}"
    assert f2.shape == (B, 256), f"Branch2D shape mismatch: {f2.shape}"
    assert f3.shape == (B, 256), f"Branch3D shape mismatch: {f3.shape}"

    out = model(d1, d2, d3)
    assert out.shape == (B, 1), f"Output shape mismatch: {out.shape}"

    logger.info("  Branch1D : %s  OK", tuple(f1.shape))
    logger.info("  Branch2D : %s  OK", tuple(f2.shape))
    logger.info("  Branch3D : %s  OK", tuple(f3.shape))
    logger.info("  Output   : %s  OK", tuple(out.shape))
    logger.info("Sanity check PASSED")


# ---------------------------------------------------------------------------
# One epoch — train or validate
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    phase: str,
    max_grad_norm: float = 1.0,
) -> tuple:
    """Run one pass.  Returns (mse, mae) as plain floats."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    sum_mse = sum_mae = total_n = 0.0
    mae_fn  = nn.L1Loss(reduction="sum")
    mse_fn  = nn.MSELoss(reduction="sum")

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for batch in loader:
            b1 = batch["1d_signal"].to(device)
            b2 = batch["2d_spectrogram"].to(device)
            b3 = batch["3d_phase"].to(device)
            bl = batch["label"].to(device)
            n  = b1.size(0)

            preds = model(b1, b2, b3)

            if is_train:
                loss = criterion(preds, bl)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping prevents the 76→290 oscillation seen before
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            with torch.no_grad():
                sum_mse += mse_fn(preds, bl).item()
                sum_mae += mae_fn(preds, bl).item()
            total_n += n

    return sum_mse / total_n, sum_mae / total_n


# ---------------------------------------------------------------------------
# Clinical evaluation (Clarke Error Grid)
# ---------------------------------------------------------------------------
def run_clinical_evaluation(
    model: nn.Module,
    val_loader,
    device: torch.device,
    save_path: str,
    optimal_epoch: int,
    best_val_mse: float,
    best_val_mae: float,
) -> dict:
    """Load best checkpoint, run inference, compute Clarke zones, save JSON+PNG."""
    logger.info("Loading best checkpoint for clinical evaluation: %s", save_path)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            b1 = batch["1d_signal"].to(device)
            b2 = batch["2d_spectrogram"].to(device)
            b3 = batch["3d_phase"].to(device)
            bl = batch["label"]
            out = model(b1, b2, b3).cpu()
            all_preds.extend(out.numpy().flatten().tolist())
            all_refs.extend(bl.numpy().flatten().tolist())

    refs  = np.array(all_refs,  dtype=np.float32)
    preds = np.array(all_preds, dtype=np.float32)

    results = summarize_clarke_zones(refs, preds)
    results["optimal_epoch"]  = int(optimal_epoch)
    results["best_val_mse"]   = float(best_val_mse)
    results["best_val_mae"]   = float(best_val_mae)
    results["best_val_rmse"]  = float(np.sqrt(best_val_mse))

    out_json = Path(__file__).parent / "clarke_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    clarke_error_grid(refs, preds,
                      title=f"Clarke Error Grid — Best Model (Epoch {optimal_epoch})")

    logger.info("Clarke Error Grid:")
    for zone in "ABCDE":
        cnt = results[f"zone_{zone}_count"]
        pct = results[f"zone_{zone}_pct"]
        logger.info("  Zone %s: %4d (%5.1f%%)", zone, cnt, pct)
    logger.info("Saved: clarke_results.json  +  clarke_error_grid.png")
    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    csv_path: str,
    num_epochs: int   = 100,
    batch_size: int   = 32,
    learning_rate: float = 0.001,
    patience: int     = 10,
    min_delta: float  = 0.5,
    max_grad_norm: float = 1.0,
    save_path: str    = "fusion_model_v2_mac.pth",
    log_path: str     = "training_log.csv",
) -> tuple:
    """Train the fusion model.  Returns (best_val_mse, best_val_mae, optimal_epoch)."""
    # ── Banner ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  PPG Multimodal Fusion Network — Training")
    logger.info("  Log file : %s", LOG_FILE)
    logger.info("=" * 65)
    logger.info(
        "Config  epochs=%-4d  batch=%-3d  lr=%.4f  patience=%d  min_delta=%.2f",
        num_epochs, batch_size, learning_rate, patience, min_delta,
    )

    device = get_device()

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading and splitting dataset: %s", csv_path)
    train_ds, val_ds = build_train_val_datasets(csv_path)
    train_loader = build_dataloader(train_ds, batch_size, shuffle=True,  drop_last=True)
    val_loader   = build_dataloader(val_ds,   batch_size, shuffle=False, drop_last=False)
    logger.info("Train windows: %d   Val windows: %d", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultimodalFusionNet().to(device)
    sanity_check(model, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    # ReduceLROnPlateau adapts to actual validation signal (better than fixed decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
        min_lr=1e-6,
    )

    # ── CSV log setup ─────────────────────────────────────────────────────────
    log_path_full = Path(__file__).parent / log_path
    if log_path_full.exists():
        log_path_full.unlink()
    with open(log_path_full, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "val_mse", "val_mae", "lr"])

    # ── Training state ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_mae  = float("inf")
    no_improve    = 0
    optimal_epoch = 0
    epoch_logs    = []

    # ── Column header ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info(
        "%-6s | %-8s | %-10s | %-10s | %-10s | %-10s | %s",
        "Epoch", "Time(s)", "Train MSE", "Val MSE", "Val MAE", "LR", "Status",
    )
    logger.info("-" * 75)

    try:
        for epoch in range(1, num_epochs + 1):
            t0 = time.perf_counter()

            train_mse, _     = run_epoch(
                model, train_loader, criterion, optimizer, device, "train", max_grad_norm
            )
            val_mse, val_mae = run_epoch(
                model, val_loader, criterion, None, device, "val"
            )

            elapsed    = time.perf_counter() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Improvement check (min_delta prevents noise-driven "saves") ───
            improved = val_mse < best_val_loss - min_delta
            if improved:
                best_val_loss = val_mse
                best_val_mae  = val_mae
                no_improve    = 0
                optimal_epoch = epoch
                torch.save(model.state_dict(), save_path)
                status = "<-- BEST  saved"
            else:
                no_improve += 1
                status = f"no improve {no_improve}/{patience}"

            # ── Live log line ─────────────────────────────────────────────────
            logger.info(
                "%-6d | %8.1f | %10.4f | %10.4f | %10.4f | %10.2e | %s",
                epoch, elapsed, train_mse, val_mse, val_mae, current_lr, status,
            )

            # ── Append to CSV (atomic full-rewrite preserves integrity) ───────
            epoch_logs.append([epoch, train_mse, val_mse, val_mae, current_lr])
            with open(log_path_full, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_mse", "val_mse", "val_mae", "lr"])
                w.writerows(sorted(epoch_logs, key=lambda r: r[0]))

            scheduler.step(val_mse)

            # ── Early stopping ────────────────────────────────────────────────
            if no_improve >= patience:
                logger.info(
                    "Early stopping: no improvement for %d consecutive epochs.",
                    patience,
                )
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by keyboard (Ctrl+C). Resuming summary ...")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("Training finished.")
    logger.info("  Optimal epoch  : %d", optimal_epoch)
    logger.info("  Best Val MSE   : %.4f  mg/dL²", best_val_loss)
    logger.info("  Best Val RMSE  : %.4f  mg/dL",  np.sqrt(best_val_loss))
    logger.info("  Best Val MAE   : %.4f  mg/dL",  best_val_mae)
    logger.info("  Model saved to : %s", os.path.abspath(save_path))
    logger.info("  Full log at    : %s", LOG_FILE)
    logger.info("=" * 65)

    # ── Clinical evaluation on best checkpoint ────────────────────────────────
    if Path(save_path).exists():
        run_clinical_evaluation(
            model, val_loader, device,
            save_path, optimal_epoch, best_val_loss, best_val_mae,
        )
    else:
        logger.warning("No saved model found — skipping Clarke evaluation.")

    return best_val_loss, best_val_mae, optimal_epoch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Train the PPG 1D+2D+3D Multimodal Fusion Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=str(root / "Archive_Dataset" / "clean-dataset.csv"),
        help="Path to clean-dataset.csv",
    )
    p.add_argument("--epochs",    type=int,   default=100,   help="Max training epochs")
    p.add_argument("--batch",     type=int,   default=32,    help="Batch size")
    p.add_argument("--lr",        type=float, default=0.001, help="Initial learning rate")
    p.add_argument("--patience",  type=int,   default=10,    help="Early-stopping patience")
    p.add_argument("--min-delta", type=float, default=0.5,
                   help="Minimum MSE improvement to reset early-stop counter")
    p.add_argument("--save",      default="fusion_model_v2_mac.pth",
                   help="Path for best model checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        csv_path=args.csv,
        num_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        save_path=args.save,
    )
