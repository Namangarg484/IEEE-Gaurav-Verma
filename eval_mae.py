"""
eval_mae.py
===========
Comprehensive evaluation of the trained Multimodal Fusion model.

Metrics reported
----------------
    MSE   — Mean Squared Error (mg/dL²)
    RMSE  — Root Mean Squared Error (mg/dL)
    MAE   — Mean Absolute Error (mg/dL)
    R²    — Coefficient of determination
    MARD  — Mean Absolute Relative Difference (%)
    Clarke Error Grid zone distribution (A/B/C/D/E)

Usage
-----
    python eval_mae.py
    python eval_mae.py --model fusion_model_v2_mac.pth
    python eval_mae.py --csv /path/to/clean-dataset.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fusion_data_pipeline import build_dataloader, build_train_val_datasets
from fusion_network_model import MultimodalFusionNet
from plot_clarke_error_grid import summarize_clarke_zones
from train_fusion_model import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_metrics(refs: np.ndarray, preds: np.ndarray) -> dict:
    """Compute MSE, RMSE, MAE, R², MARD from reference and predicted arrays."""
    mse  = float(np.mean((preds - refs) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(preds - refs)))

    ss_res = float(np.sum((refs - preds) ** 2))
    ss_tot = float(np.sum((refs - refs.mean()) ** 2))
    r2     = float(1.0 - ss_res / (ss_tot + 1e-8))

    # MARD: mean(|pred - ref| / ref) × 100
    mard = float(np.mean(np.abs(preds - refs) / (np.abs(refs) + 1e-8)) * 100.0)

    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, mard_pct=mard)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: nn.Module, loader, device: torch.device
) -> tuple:
    """Run model over a DataLoader. Returns (refs, preds) as numpy arrays."""
    model.eval()
    preds_all, refs_all = [], []
    with torch.no_grad():
        for batch in loader:
            b1  = batch["1d_signal"].to(device)
            b2  = batch["2d_spectrogram"].to(device)
            b3  = batch["3d_phase"].to(device)
            bl  = batch["label"]
            out = model(b1, b2, b3).cpu().numpy().flatten()
            preds_all.extend(out.tolist())
            refs_all.extend(bl.numpy().flatten().tolist())
    return np.array(refs_all, dtype=np.float32), np.array(preds_all, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate(csv_path: str, model_path: str) -> dict:
    """Full evaluation pipeline. Returns results dict."""
    logger.info("=" * 58)
    logger.info("  PPG Fusion Network — Evaluation Report")
    logger.info("=" * 58)

    device = get_device()

    # ── Load validation data ──────────────────────────────────────────────────
    logger.info("Building validation dataset from: %s", csv_path)
    _, val_ds = build_train_val_datasets(csv_path)
    val_loader = build_dataloader(val_ds, batch_size=32, shuffle=False)
    logger.info("Validation windows: %d", len(val_ds))

    # ── Load model ────────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        logger.error("Model checkpoint not found: %s", model_path)
        sys.exit(1)
    model = MultimodalFusionNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info("Loaded checkpoint: %s", model_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference ...")
    refs, preds = run_inference(model, val_loader, device)
    logger.info("Inference complete — %d predictions", len(refs))

    # ── Global metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(refs, preds)
    logger.info("")
    logger.info("Global Metrics  (n=%d):", len(refs))
    logger.info("  MSE   : %9.4f  mg/dL²", metrics["mse"])
    logger.info("  RMSE  : %9.4f  mg/dL",  metrics["rmse"])
    logger.info("  MAE   : %9.4f  mg/dL",  metrics["mae"])
    logger.info("  R²    : %9.4f",          metrics["r2"])
    logger.info("  MARD  : %9.2f  %%",      metrics["mard_pct"])

    # ── Clarke Error Grid ─────────────────────────────────────────────────────
    zones = summarize_clarke_zones(refs, preds)
    logger.info("")
    logger.info("Clarke Error Grid:")
    for z in "ABCDE":
        cnt = zones[f"zone_{z}_count"]
        pct = zones[f"zone_{z}_pct"]
        bar = "#" * int(pct / 2)
        logger.info("  Zone %s: %4d  (%5.1f%%)  %s", z, cnt, pct, bar)

    # ── Glucose range in validation set ──────────────────────────────────────
    logger.info("")
    logger.info("Validation glucose range: %.1f – %.1f mg/dL",
                refs.min(), refs.max())
    logger.info("Prediction range        : %.1f – %.1f mg/dL",
                preds.min(), preds.max())

    # ── Save results ──────────────────────────────────────────────────────────
    results  = {**metrics, **zones}
    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("Results saved to: %s", out_path)
    logger.info("=" * 58)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    root = Path(__file__).parent
    p = argparse.ArgumentParser(
        description="Evaluate the PPG Fusion Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default=str(root / "Archive_Dataset" / "clean-dataset.csv"),
    )
    p.add_argument(
        "--model",
        default=str(root / "fusion_model_v2_mac.pth"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    evaluate(args.csv, args.model)
