"""
linear_probe.py
===============
Cross-dataset generalisation experiment using the pretrained MultimodalFusionNet.

Strategy (linear probe / fine-tune hybrid)
-------------------------------------------
  • All three branch encoders (1D-CNN, 2D-CNN, PointNet) are FROZEN.
  • Only the CrossAttentionFusion layer (module name: 'fusion') and
    the FC regression head (module name: 'head') are fine-tuned.
    (These correspond to the 'attention', 'fc_head', 'regression' semantic
    roles specified in the experiment design.)
  • On-the-fly feature generation reuses the exact same CWT + Takens
    transforms as the original training pipeline (imported from
    fusion_data_pipeline.py) to avoid any preprocessing mismatch.

Input files (created by prepare_generalization_data.py):
  physionet_X_500.npy       shape (N, 500)  float32 resampled PPG windows
  physionet_y_glucose.npy   shape (N,)      float32 glucose in mg/dL

Output (printed):
  Cross-Dataset Generalisation MAE  (mg/dL)
  Cross-Dataset Generalisation RMSE (mg/dL)

Usage:
  python linear_probe.py
  python linear_probe.py --epochs 20 --batch-size 64 --lr 5e-4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Local imports — must be run from workspace root
from fusion_network_model import MultimodalFusionNet
from fusion_data_pipeline  import PPGFusionDataset   # for CWT & phase-space helpers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers for feature transforms (same as training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
_cwt_scalogram = PPGFusionDataset._cwt_scalogram   # (1D np) → np (1,64,64)
_phase_space   = PPGFusionDataset._phase_space     # (1D np, tau, n) → np (3,N)

TAU         = 5
PHASE_POINTS = 256


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PhysioNetProbeDataset(Dataset):
    """Wraps resampled 1-D PPG windows; generates all three modalities on-the-fly.

    __getitem__ mirrors PPGFusionDataset exactly:
      x_1d   FloatTensor (1, 500)   — z-score normalised signal
      x_2d   FloatTensor (1,64,64)  — CWT scalogram (Mexican Hat)
      x_3d   FloatTensor (3,256)    — Takens delay embedding
      label  FloatTensor (1,)       — glucose in mg/dL
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        assert signals.ndim == 2 and signals.shape[1] == 500, (
            f"Expected signals shape (N, 500), got {signals.shape}"
        )
        assert len(signals) == len(labels), "signals / labels length mismatch"
        self.signals = signals.astype(np.float32)
        self.labels  = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        sig = self.signals[idx]                             # (500,)

        # Z-score normalise (same as PPGFusionDataset)
        mu, sigma = float(sig.mean()), float(sig.std())
        sig_norm  = (sig - mu) / (sigma + 1e-8)

        x_1d = torch.from_numpy(sig_norm[np.newaxis, :])                    # (1,500)
        x_2d = torch.from_numpy(_cwt_scalogram(sig_norm))                   # (1,64,64)
        x_3d = torch.from_numpy(_phase_space(sig_norm, TAU, PHASE_POINTS))  # (3,256)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)        # (1,)

        return x_1d, x_2d, x_3d, label


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device) -> MultimodalFusionNet:
    """Load pretrained MultimodalFusionNet from flat-key checkpoint."""
    model = MultimodalFusionNet()

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Handle both bare state-dict and {'model_state_dict': ...} wrappers
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys in checkpoint (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        log.warning("Unexpected keys in checkpoint (%d): %s ...", len(unexpected), unexpected[:3])
    log.info("Checkpoint loaded from: %s", ckpt_path)

    model.to(device)
    return model


def apply_freeze(model: MultimodalFusionNet) -> int:
    """Freeze all params EXCEPT those in 'fusion' (cross-attention) and 'head' (FC).

    Module mapping to user spec:
      'fusion' ≡ CrossAttentionFusion  ('attention' in user spec)
      'head'   ≡ FC regression stack   ('fc_head' / 'regression' in user spec)

    Returns the number of trainable parameters after freezing.
    """
    TRAINABLE_PREFIXES = ("fusion.", "head.")

    total_params    = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        is_trainable = any(name.startswith(p) for p in TRAINABLE_PREFIXES)
        param.requires_grad = is_trainable
        total_params     += param.numel()
        if is_trainable:
            trainable_params += param.numel()

    frozen = total_params - trainable_params
    log.info(
        "Parameter freeze applied:"
        "\n  Total params    : %d"
        "\n  Frozen          : %d  (branches 1D, 2D, 3D)"
        "\n  Trainable       : %d  (cross-attention fusion + FC head)",
        total_params, frozen, trainable_params,
    )
    return trainable_params


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model: MultimodalFusionNet,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x_1d, x_2d, x_3d, labels in loader:
        x_1d   = x_1d.to(device)
        x_2d   = x_2d.to(device)
        x_3d   = x_3d.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        preds = model(x_1d, x_2d, x_3d)           # (B, 1)
        loss  = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        optimiser.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: MultimodalFusionNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (MAE, RMSE) in mg/dL."""
    model.eval()
    all_preds, all_labels = [], []
    for x_1d, x_2d, x_3d, labels in loader:
        preds = model(x_1d.to(device), x_2d.to(device), x_3d.to(device))
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    preds_np  = torch.cat(all_preds).squeeze().numpy()
    labels_np = torch.cat(all_labels).squeeze().numpy()

    mae  = float(np.mean(np.abs(preds_np - labels_np)))
    rmse = float(np.sqrt(np.mean((preds_np - labels_np) ** 2)))
    return mae, rmse


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset generalisation linear probe on PhysioNet data"
    )
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint (default: fusion_model_v2_mac.pth alongside script)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing physionet_X_500.npy / physionet_y_glucose.npy",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir   = Path(args.data_dir) if args.data_dir else script_dir
    ckpt_path  = Path(args.ckpt)     if args.ckpt    else script_dir / "fusion_model_v2_mac.pth"

    x_path = data_dir / "physionet_X_500.npy"
    y_path = data_dir / "physionet_y_glucose.npy"

    for p in (x_path, y_path, ckpt_path):
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: CUDA GPU")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    # ── Data ─────────────────────────────────────────────────────────────────
    X = np.load(x_path)
    y = np.load(y_path)
    log.info("Loaded X=%s  y=%s", X.shape, y.shape)
    log.info("Glucose: %.1f – %.1f mg/dL  (mean %.1f, std %.1f)",
             y.min(), y.max(), y.mean(), y.std())

    full_dataset = PhysioNetProbeDataset(X, y)

    n_train = int(len(full_dataset) * args.train_frac)
    n_test  = len(full_dataset) - n_train
    train_ds, test_ds = random_split(
        full_dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    log.info("Train: %d  |  Test: %d  (%.0f/%.0f split)",
             n_train, n_test, args.train_frac * 100, (1 - args.train_frac) * 100)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)
    apply_freeze(model)

    # Optimise only the trainable (unfrozen) parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.MSELoss()

    # ── Baseline (no fine-tuning) ─────────────────────────────────────────────
    log.info("\nEvaluating zero-shot baseline (frozen model, no fine-tuning) ...")
    baseline_mae, baseline_rmse = evaluate(model, test_loader, device)
    log.info("  Zero-shot MAE : %.2f mg/dL", baseline_mae)
    log.info("  Zero-shot RMSE: %.2f mg/dL", baseline_rmse)

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    log.info("\nFine-tuning (epochs=%d, lr=%g, batch=%d) ...\n",
             args.epochs, args.lr, args.batch_size)

    for epoch in range(1, args.epochs + 1):
        t0        = time.time()
        train_mse = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_mae, val_rmse = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        log.info(
            "Epoch %2d/%2d | TrainMSE: %8.2f | Val MAE: %6.2f | Val RMSE: %6.2f | %.1f s",
            epoch, args.epochs,
            train_mse, val_mae, val_rmse, elapsed,
        )

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_mae, final_rmse = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print(" Cross-Dataset Generalisation Results (PhysioNet BIG IDEAs)")
    print("=" * 60)
    print(f"  Zero-shot MAE  (before fine-tuning): {baseline_mae:.2f} mg/dL")
    print(f"  Zero-shot RMSE (before fine-tuning): {baseline_rmse:.2f} mg/dL")
    print(f"  Fine-tuned MAE  (after {args.epochs} epochs):  {final_mae:.2f} mg/dL")
    print(f"  Fine-tuned RMSE (after {args.epochs} epochs):  {final_rmse:.2f} mg/dL")
    print(f"  Improvement:  ΔMAE = {baseline_mae - final_mae:+.2f} mg/dL")
    print("=" * 60)


if __name__ == "__main__":
    main()
