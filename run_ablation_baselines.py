#!/usr/bin/env python3
"""
run_ablation_baselines.py
=========================
Implements and evaluates ALL baselines required for a rigorous ablation study.

Models evaluated
----------------
  Sanity Baselines (no training):
    1.  Global Mean Predictor       – predict training-set mean for every window
    2.  Per-Patient Mean Predictor  – predict that patient's own val-set mean
    3.  Vitals Linear Regression    – Ridge on HR, Systolic/Diastolic, Pulse_Area,
                                      Age, Gender, BMI (window-aggregated)

  Modality Ablation (trained from scratch, same protocol):
    4.  1D-only CNN
    5.  2D-only CNN
    6.  3D-only PointNet
    7.  1D+2D (concatenation)
    8.  1D+3D (concatenation)
    9.  1D+2D+3D concat (no attention)
   10.  1D+2D+3D cross-attention  ← load from fusion_model_v2_mac.pth

  SOTA Re-implementations:
   11.  ResNet-1D
   12.  CNN-LSTM

Outputs
-------
    ablation_results.json        — all metrics (MAE / RMSE / ZoneA)
    Paper_Table2_Ablation.png    — horizontal bar-chart comparison
    checkpoints/ablation_*/      — saved .pth for each trained model

Usage
-----
    python run_ablation_baselines.py [--epochs 100] [--batch 16] [--device auto]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fusion_data_pipeline import (
    WINDOW_SIZE,
    PPGFusionDataset,
    build_dataloader,
    build_train_val_datasets,
)
from fusion_network_model import Branch1D, Branch2D, Branch3D, MultimodalFusionNet

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = "cpu"          # set in main()
CSV    = Path("Archive_Dataset/clean-dataset.csv")
CKPT   = Path("fusion_model_v2_mac.pth")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Ablation model architectures
# ─────────────────────────────────────────────────────────────────────────────

class AblationModel(nn.Module):
    """Parameterised model using any subset of the three branches.
    Branches whose flag is False are never instantiated (saves memory).
    Fusion: simple concatenation of branch outputs (no attention).
    """

    def __init__(
        self,
        use_1d: bool = True,
        use_2d: bool = True,
        use_3d: bool = True,
    ) -> None:
        super().__init__()
        assert any([use_1d, use_2d, use_3d]), "At least one branch required."
        self.use_1d = use_1d
        self.use_2d = use_2d
        self.use_3d = use_3d

        if use_1d:
            self.branch1d = Branch1D()    # → 256
        if use_2d:
            self.branch2d = Branch2D()    # → 256
        if use_3d:
            self.branch3d = Branch3D()    # → 256

        n_branches = sum([use_1d, use_2d, use_3d])
        feat_dim   = n_branches * 256

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        feats = []
        if self.use_1d:
            feats.append(self.branch1d(batch["1d_signal"].to(DEVICE)))
        if self.use_2d:
            feats.append(self.branch2d(batch["2d_spectrogram"].to(DEVICE)))
        if self.use_3d:
            feats.append(self.branch3d(batch["3d_phase"].to(DEVICE)))
        return self.head(torch.cat(feats, dim=-1))


# ── ResNet-1D ─────────────────────────────────────────────────────────────────

class _ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """Standard 1D ResNet for time-series regression.
    Input: batch dict → uses '1d_signal' (B,1,500).
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )   # (B,64,125)
        self.layer1 = _ResBlock1D(64,  64)
        self.layer2 = _ResBlock1D(64,  128, stride=2)
        self.layer3 = _ResBlock1D(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["1d_signal"].to(DEVICE)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# ── CNN-LSTM ──────────────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """Convolutional feature extractor followed by a 2-layer LSTM.
    Input: batch dict → uses '1d_signal' (B,1,500).
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,  32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )   # (B, 128, 125)
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128,
            num_layers=2,   batch_first=True,
            dropout=0.2,    bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["1d_signal"].to(DEVICE)          # (B,1,500)
        x = self.cnn(x)                             # (B,128,T')
        x = x.permute(0, 2, 1)                      # (B,T',128)
        _, (h_n, _) = self.lstm(x)                  # h_n: (2,B,128)
        out = h_n[-1]                                # (B,128) last layer
        return self.head(out)


# ─────────────────────────────────────────────────────────────────────────────
# Generic training & evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clarke_zone_a(preds: np.ndarray, refs: np.ndarray) -> float:
    """Fraction of (ref, pred) pairs in Clarke Error Grid Zone A."""
    n = len(refs)
    in_a = 0
    for ref, pred in zip(refs, preds):
        # Absolute difference within 20% of ref
        if abs(pred - ref) / max(ref, 1e-6) <= 0.20:
            in_a += 1
        # Both below 70 (hypoglycaemic zone)
        elif ref < 70 and pred < 70:
            in_a += 1
    return in_a / n


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    name: str,
    max_epochs: int = 100,
    lr: float = 5e-4,
    patience: int = 15,
) -> dict:
    """Generic training loop.  Returns dict with training stats and best metrics."""

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    train_losses  = []
    val_losses    = []

    t0 = time.time()
    log.info("  Training %-28s | epochs≤%d | device=%s", name, max_epochs, DEVICE)

    for epoch in range(1, max_epochs + 1):
        # ── train ──
        model.train()
        batch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch["label"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_mse = float(np.mean(batch_losses))

        # ── val ──
        model.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                out  = model(batch)
                all_preds.extend(out.squeeze(-1).cpu().numpy())
                all_refs.extend(batch["label"].squeeze(-1).cpu().numpy())

        preds = np.array(all_preds)
        refs  = np.array(all_refs)
        val_mse = float(np.mean((preds - refs) ** 2))
        val_mae = float(np.mean(np.abs(preds - refs)))

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        scheduler.step(val_mse)

        # Progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "  [%s] Epoch %3d  train_mse=%.2f  val_mse=%.2f  val_mae=%.3f",
                name, epoch, train_mse, val_mse, val_mae,
            )

        # Early stopping
        if val_mse < best_val_loss - 0.5:
            best_val_loss  = val_mse
            best_epoch     = epoch
            no_improve     = 0
            ckpt_path = CKPT_DIR / f"ablation_{name.replace(' ', '_').lower()}.pth"
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  [%s] Early stop at epoch %d (best=%d)", name, epoch, best_epoch)
                break

    elapsed = time.time() - t0

    # Reload best weights
    ckpt_path = CKPT_DIR / f"ablation_{name.replace(' ', '_').lower()}.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEVICE))

    # Final evaluation
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            all_preds.extend(out.squeeze(-1).cpu().numpy())
            all_refs.extend(batch["label"].squeeze(-1).cpu().numpy())

    preds = np.array(all_preds)
    refs  = np.array(all_refs)
    mae   = float(np.mean(np.abs(preds - refs)))
    rmse  = float(np.sqrt(np.mean((preds - refs) ** 2)))
    za    = float(_clarke_zone_a(preds, refs)) * 100

    log.info(
        "  [%s] DONE  MAE=%.3f  RMSE=%.3f  ZoneA=%.1f%%  time=%.0fs",
        name, mae, rmse, za, elapsed,
    )
    return {
        "name":        name,
        "mae":         mae,
        "rmse":        rmse,
        "zone_a_pct":  za,
        "best_epoch":  best_epoch,
        "train_time_s": elapsed,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
    }


def eval_only(
    model: nn.Module,
    val_loader,
    name: str,
    ckpt_path: Path | None = None,
) -> dict:
    """Evaluate a pre-loaded (or checkpoint-provided) model."""
    if ckpt_path and ckpt_path.exists():
        model.load_state_dict(
            torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
        )
    model = model.to(DEVICE)
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            all_preds.extend(out.squeeze(-1).cpu().numpy())
            all_refs.extend(batch["label"].squeeze(-1).cpu().numpy())
    preds = np.array(all_preds)
    refs  = np.array(all_refs)
    mae   = float(np.mean(np.abs(preds - refs)))
    rmse  = float(np.sqrt(np.mean((preds - refs) ** 2)))
    za    = float(_clarke_zone_a(preds, refs)) * 100
    log.info("  [%s] MAE=%.3f  RMSE=%.3f  ZoneA=%.1f%%", name, mae, rmse, za)
    return {
        "name": name, "mae": mae, "rmse": rmse,
        "zone_a_pct": za, "best_epoch": "pretrained",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity baselines (no neural network required)
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_baselines(df: pd.DataFrame, train_ids, val_ids) -> list:
    """Compute mean predictors and vitals regression without any neural model."""
    results = []

    train_df = df[df["Patient_Id"].isin(train_ids)].copy()
    val_df   = df[df["Patient_Id"].isin(val_ids)].copy()

    # Build window-level tables
    def window_table(frame):
        rows = []
        n_windows = len(frame) // WINDOW_SIZE
        for i in range(n_windows):
            chunk = frame.iloc[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
            row = {"glucose": chunk["Glucose_level"].mean(),
                   "patient": int(chunk["Patient_Id"].iloc[0])}
            for col in ["Heart_Rate", "Systolic_Peak", "Diastolic_Peak", "Pulse_Area",
                        "Age", "Gender", "Height", "Weight"]:
                if col in chunk.columns:
                    row[col] = float(chunk[col].mean())
            rows.append(row)
        return pd.DataFrame(rows)

    train_win = window_table(train_df)
    val_win   = window_table(val_df)

    # 1. Global mean predictor
    global_mean = float(train_win["glucose"].mean())
    preds_global = np.full(len(val_win), global_mean)
    refs = val_win["glucose"].values
    mae  = float(np.mean(np.abs(preds_global - refs)))
    rmse = float(np.sqrt(np.mean((preds_global - refs) ** 2)))
    za   = _clarke_zone_a(preds_global, refs) * 100
    log.info("  [Global Mean] MAE=%.3f  RMSE=%.3f  ZoneA=%.1f%%", mae, rmse, za)
    results.append({"name": "Global Mean Predictor",
                    "mae": mae, "rmse": rmse, "zone_a_pct": za,
                    "best_epoch": "N/A"})

    # 2. Per-patient mean predictor (oracle: predicts each patient's own mean)
    per_patient_mean = val_win.groupby("patient")["glucose"].transform("mean").values
    mae  = float(np.mean(np.abs(per_patient_mean - refs)))
    rmse = float(np.sqrt(np.mean((per_patient_mean - refs) ** 2)))
    za   = _clarke_zone_a(per_patient_mean, refs) * 100
    log.info("  [Per-Patient Mean] MAE=%.3f  RMSE=%.3f  ZoneA=%.1f%%", mae, rmse, za)
    results.append({"name": "Per-Patient Mean (Oracle)",
                    "mae": mae, "rmse": rmse, "zone_a_pct": za,
                    "best_epoch": "N/A"})

    # 3. Vitals linear regression
    feature_cols = [c for c in ["Heart_Rate", "Systolic_Peak", "Diastolic_Peak",
                                 "Pulse_Area", "Age", "Gender"]
                    if c in train_win.columns]
    if len(feature_cols) >= 2:
        X_tr = train_win[feature_cols].fillna(0).values
        y_tr = train_win["glucose"].values
        X_va = val_win[feature_cols].fillna(0).values
        y_va = refs

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # Remove any constant (zero-variance) features to avoid overflow
        nonconst = X_tr_s.std(axis=0) > 1e-8
        X_tr_s   = X_tr_s[:, nonconst]
        X_va_s   = X_va_s[:, nonconst]

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_s, y_tr)
        preds_ridge = np.clip(ridge.predict(X_va_s), 50, 300)  # physiological clamp

        mae  = float(np.mean(np.abs(preds_ridge - y_va)))
        rmse = float(np.sqrt(np.mean((preds_ridge - y_va) ** 2)))
        za   = _clarke_zone_a(preds_ridge, y_va) * 100
        log.info("  [Vitals Ridge] MAE=%.3f  RMSE=%.3f  ZoneA=%.1f%%  features=%s",
                 mae, rmse, za, feature_cols)
        results.append({"name": "Vitals Ridge Regression",
                        "mae": mae, "rmse": rmse, "zone_a_pct": za,
                        "best_epoch": "N/A", "features": feature_cols})
    else:
        log.warning("  Not enough feature columns for vitals regression: %s", feature_cols)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_figure(results: list, save_path: str = "Paper_Table2_Ablation.png") -> None:
    """Horizontal bar chart comparing all methods by MAE."""

    names = [r["name"] for r in results]
    maes  = [r["mae"]  for r in results]
    zones = [r.get("zone_a_pct", 0) for r in results]

    # Sort by MAE ascending
    order  = np.argsort(maes)
    names  = [names[i]  for i in order]
    maes   = [maes[i]   for i in order]
    zones  = [zones[i]  for i in order]

    colors = []
    our_name = "1D+2D+3D Cross-Attention (Ours)"
    for n in names:
        if "Ours" in n or "Cross-Attention" in n:
            colors.append("#1a73e8")
        elif "1D+2D+3D" in n:
            colors.append("#34a853")
        elif "ResNet" in n or "CNN-LSTM" in n:
            colors.append("#ea4335")
        elif "Mean" in n or "Ridge" in n or "Oracle" in n:
            colors.append("#9e9e9e")
        else:
            colors.append("#fbbc04")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(names) * 0.55)))

    y_pos = np.arange(len(names))

    # MAE bars
    bars = ax1.barh(y_pos, maes, color=colors, edgecolor="white", height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("MAE (mg/dL)", fontsize=11)
    ax1.set_title("Validation MAE (lower is better)", fontsize=12, fontweight="bold")
    ax1.axvline(x=6.89, color="#1a73e8", linestyle="--", linewidth=1.5, alpha=0.6,
                label="Our model (6.89)")
    for i, (bar, v) in enumerate(zip(bars, maes)):
        ax1.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, max(maes) * 1.18)
    ax1.invert_yaxis()

    # Zone A bars
    bars2 = ax2.barh(y_pos, zones, color=colors, edgecolor="white", height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Clarke Zone A (%)", fontsize=11)
    ax2.set_title("Clarke Zone A (higher is better)", fontsize=12, fontweight="bold")
    ax2.axvline(x=95.0, color="red", linestyle="--", linewidth=1.5, alpha=0.6,
                label="ISO 15197 threshold (95%)")
    ax2.axvline(x=98.9, color="#1a73e8", linestyle="--", linewidth=1.5, alpha=0.6,
                label="Our model (98.9%)")
    for i, (bar, v) in enumerate(zip(bars2, zones)):
        ax2.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 105)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved ablation figure: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--device",  type=str,   default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--skip-trained", action="store_true",
                   help="Skip models whose checkpoint already exists.")
    return p.parse_args()


def main():
    global DEVICE
    args = parse_args()

    # Device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            DEVICE = "mps"
        elif torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"
    else:
        DEVICE = args.device
    log.info("Device: %s", DEVICE)

    if not CSV.exists():
        sys.exit(f"ERROR: CSV not found: {CSV}")

    # Load full dataframe for sanity baselines
    log.info("Loading dataset …")
    df       = pd.read_csv(str(CSV))
    patients = df["Patient_Id"].unique()
    train_ids = patients[:18]
    val_ids   = patients[18:]

    train_ds, val_ds = build_train_val_datasets(str(CSV))
    train_loader = build_dataloader(train_ds, batch_size=args.batch,
                                    shuffle=True, drop_last=True)
    val_loader   = build_dataloader(val_ds, batch_size=64,
                                    shuffle=False, drop_last=False)

    log.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))

    all_results = []

    # ── 1. Sanity baselines ───────────────────────────────────────────────────
    log.info("\n=== SANITY BASELINES ===")
    sanity = run_sanity_baselines(df, train_ids, val_ids)
    all_results.extend(sanity)

    # ── 2. Modality ablation ──────────────────────────────────────────────────
    log.info("\n=== MODALITY ABLATION ===")
    ablations = [
        ("1D-Only",         AblationModel(use_1d=True,  use_2d=False, use_3d=False)),
        ("2D-Only",         AblationModel(use_1d=False, use_2d=True,  use_3d=False)),
        ("3D-Only",         AblationModel(use_1d=False, use_2d=False, use_3d=True)),
        ("1D+2D Concat",    AblationModel(use_1d=True,  use_2d=True,  use_3d=False)),
        ("1D+3D Concat",    AblationModel(use_1d=True,  use_2d=False, use_3d=True)),
        ("1D+2D+3D Concat", AblationModel(use_1d=True,  use_2d=True,  use_3d=True)),
    ]
    for name, model in ablations:
        ckpt_p = CKPT_DIR / f"ablation_{name.replace(' ', '_').lower()}.pth"
        if args.skip_trained and ckpt_p.exists():
            log.info("  [%s] Skipping (checkpoint exists).", name)
            r = eval_only(model, val_loader, name, ckpt_p)
        else:
            r = train_model(model, train_loader, val_loader, name,
                            max_epochs=args.epochs, patience=15)
        all_results.append(r)

    # ── 3. SOTA baselines ─────────────────────────────────────────────────────
    log.info("\n=== SOTA BASELINES ===")
    sota = [
        ("ResNet-1D",  ResNet1D()),
        ("CNN-LSTM",   CNNLSTM()),
    ]
    for name, model in sota:
        ckpt_p = CKPT_DIR / f"ablation_{name.replace(' ', '_').lower()}.pth"
        if args.skip_trained and ckpt_p.exists():
            r = eval_only(model, val_loader, name, ckpt_p)
        else:
            r = train_model(model, train_loader, val_loader, name,
                            max_epochs=args.epochs, patience=15)
        all_results.append(r)

    # ── 4. Full fusion (load pre-trained checkpoint) ──────────────────────────
    log.info("\n=== FULL FUSION MODEL (from checkpoint) ===")
    if CKPT.exists():
        full_model = MultimodalFusionNet()
        # Load checkpoint into the bare MultimodalFusionNet (no wrapper prefix)
        full_model.load_state_dict(
            torch.load(CKPT, weights_only=True, map_location=DEVICE)
        )
        full_model = full_model.to(DEVICE)
        full_model.eval()

        all_preds, all_refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = full_model(
                    batch["1d_signal"].to(DEVICE),
                    batch["2d_spectrogram"].to(DEVICE),
                    batch["3d_phase"].to(DEVICE),
                )
                all_preds.extend(out.squeeze(-1).cpu().numpy())
                all_refs.extend(batch["label"].squeeze(-1).cpu().numpy())
        preds = np.array(all_preds)
        refs  = np.array(all_refs)
        mae   = float(np.mean(np.abs(preds - refs)))
        rmse  = float(np.sqrt(np.mean((preds - refs) ** 2)))
        za    = float(_clarke_zone_a(preds, refs)) * 100
        log.info("  [Full Fusion] MAE=%.3f  RMSE=%.3f  ZoneA=%.1f%%",
                 mae, rmse, za)
        all_results.append({
            "name": "1D+2D+3D Cross-Attention (Ours)",
            "mae": mae, "rmse": rmse,
            "zone_a_pct": za, "best_epoch": "pretrained",
        })
    else:
        log.warning("Checkpoint %s not found — skipping full model eval.", CKPT)

    # ── Save results ──────────────────────────────────────────────────────────
    # Strip non-JSON-serialisable fields for the JSON dump
    json_results = []
    for r in all_results:
        d = {k: v for k, v in r.items()
             if k not in ("train_losses", "val_losses")}
        json_results.append(d)

    out_json = Path("ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(json_results, f, indent=2)
    log.info("\nSaved: %s", out_json)

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n%-35s  %8s  %8s  %8s", "Model", "MAE", "RMSE", "Zone A%")
    log.info("-" * 65)
    for r in sorted(json_results, key=lambda x: x["mae"]):
        log.info(
            "%-35s  %8.3f  %8.3f  %8.1f%%",
            r["name"], r["mae"], r["rmse"], r.get("zone_a_pct", 0),
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_ablation_figure(json_results)


if __name__ == "__main__":
    main()
