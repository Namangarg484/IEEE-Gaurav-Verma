"""
fusion_data_pipeline.py
=======================
On-the-fly PyTorch Dataset for the 1D+2D+3D Multimodal Fusion Network.

Generates 2D (CWT scalogram) and 3D (Takens delay-embedding) representations
dynamically at __getitem__ time to avoid pre-computing and storing GBs of
derived data — critical for running on memory-constrained Macs.

Expected CSV columns
--------------------
    PPG_Signal    : raw ADC amplitude
    Patient_Id    : patient identifier (used for train/val split)
    Glucose_level : regression target (mg/dL)

Data-quality notes
------------------
If PPG_Signal is constant (std ≈ 0) for a window the CWT will return a
near-zero matrix and the phase-space will collapse to a single point. A
WARNING is logged at dataset construction time so the researcher is aware.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — single source of truth for shapes used across all modules
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 500    # samples per PPG segment (~5 s at 100 Hz)
TIME_DELAY    = 5      # Takens embedding tau
PHASE_POINTS  = 256    # number of 3-D phase-space points
CWT_N_SCALES  = 64     # wavelet scale levels
SPEC_H = SPEC_W = 64   # output scalogram resolution (pixels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PPGFusionDataset(Dataset):
    """Windowed PPG dataset with on-the-fly multimodal representation.

    Each item is a dict:
        '1d_signal'      : FloatTensor (1, WINDOW_SIZE)
        '2d_spectrogram' : FloatTensor (1, SPEC_H, SPEC_W)
        '3d_phase'       : FloatTensor (3, PHASE_POINTS)
        'label'          : FloatTensor (1,) — mean glucose in window (mg/dL)
    """

    def __init__(
        self,
        csv_file: str | None = None,
        dataframe: pd.DataFrame | None = None,
        window_size: int = WINDOW_SIZE,
        time_delay: int  = TIME_DELAY,
    ) -> None:
        if dataframe is not None:
            self.df = dataframe.reset_index(drop=True)
        elif csv_file is not None:
            logger.info("Reading CSV: %s", csv_file)
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Provide either csv_file or dataframe.")

        self._validate_schema()
        self.window_size = window_size
        self.time_delay  = time_delay
        self.num_samples = len(self.df) // self.window_size

        logger.info(
            "Dataset ready: %d rows → %d windows of %d samples.",
            len(self.df), self.num_samples, self.window_size,
        )
        self._audit_signal_quality()

    # ── Schema validation ────────────────────────────────────────────────────
    def _validate_schema(self) -> None:
        required = {"PPG_Signal", "Glucose_level", "Patient_Id"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    # ── Data-quality audit ───────────────────────────────────────────────────
    def _audit_signal_quality(self, n_probe: int = 50) -> None:
        """Scan the first n_probe windows and warn if signals are constant."""
        probe      = min(n_probe, self.num_samples)
        n_constant = 0
        for i in range(probe):
            vals = self.df.iloc[
                i * self.window_size : (i + 1) * self.window_size
            ]["PPG_Signal"].values
            if float(np.std(vals)) < 1e-6:
                n_constant += 1

        if n_constant:
            logger.warning(
                "DATA QUALITY: %d/%d probed windows have a CONSTANT PPG_Signal "
                "(std≈0). CWT scalograms and phase-space attractors will be "
                "uninformative for these windows. Inspect the CSV carefully.",
                n_constant, probe,
            )
        else:
            logger.info("Signal quality audit: all %d probed windows have variance > 0.", probe)

    # ── Multimodal generators ─────────────────────────────────────────────────
    @staticmethod
    def _cwt_scalogram(signal_1d: np.ndarray) -> np.ndarray:
        """CWT (Mexican Hat / Ricker wavelet) → normalised (1, 64, 64) float32."""
        scales = np.arange(1, CWT_N_SCALES + 1, dtype=np.float64)
        coef, _ = pywt.cwt(signal_1d.astype(np.float64), scales, "mexh")
        mag = np.abs(coef).astype(np.float32)            # (64, window_size)
        # Resize to square image
        mag = cv2.resize(mag, (SPEC_W, SPEC_H), interpolation=cv2.INTER_LINEAR)
        # Min-max normalise to [0, 1] for numerical stability
        mn, mx = mag.min(), mag.max()
        if mx - mn > 1e-8:
            mag = (mag - mn) / (mx - mn)
        return mag[np.newaxis, :, :]                      # (1, 64, 64)

    @staticmethod
    def _phase_space(signal_1d: np.ndarray, tau: int, n_points: int) -> np.ndarray:
        """Takens delay embedding: [x(t), x(t+τ), x(t+2τ)] → (3, n_points) float32."""
        L = len(signal_1d) - 2 * tau
        x = signal_1d[:L]
        y = signal_1d[tau : tau + L]
        z = signal_1d[2 * tau : 2 * tau + L]
        pts = np.vstack((x, y, z)).astype(np.float32)    # (3, L)
        if pts.shape[1] >= n_points:
            pts = pts[:, :n_points]
        else:
            pad = n_points - pts.shape[1]
            pts = np.pad(pts, ((0, 0), (0, pad)), mode="constant")
        return pts                                         # (3, n_points)

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.window_size
        chunk = self.df.iloc[start : start + self.window_size]

        sig     = chunk["PPG_Signal"].values.astype(np.float32)
        glucose = float(np.mean(chunk["Glucose_level"].values))

        # Z-score normalise the 1-D signal (isolates morphology from DC offset)
        mu, sigma = float(sig.mean()), float(sig.std())
        sig_norm  = (sig - mu) / (sigma + 1e-8)

        data_1d = sig_norm[np.newaxis, :]                                  # (1, 500)
        data_2d = self._cwt_scalogram(sig_norm)                            # (1, 64, 64)
        data_3d = self._phase_space(sig_norm, self.time_delay, PHASE_POINTS)  # (3, 256)

        return {
            "1d_signal":      torch.from_numpy(data_1d),
            "2d_spectrogram": torch.from_numpy(data_2d),
            "3d_phase":       torch.from_numpy(data_3d),
            "label":          torch.tensor([glucose], dtype=torch.float32),
        }

    def __repr__(self) -> str:
        return (
            f"PPGFusionDataset(samples={self.num_samples}, "
            f"window={self.window_size}, tau={self.time_delay})"
        )


# ---------------------------------------------------------------------------
# Factory helpers (used by train, eval, figure scripts)
# ---------------------------------------------------------------------------

def build_train_val_datasets(
    csv_path: str,
    window_size: int = WINDOW_SIZE,
) -> tuple:
    """Patient-level split: first 18 patients → train, rest → val.

    Returns:
        (train_dataset, val_dataset)

    Raises:
        ValueError  if fewer than 5 unique patients are found.
        AssertionError  if any patient appears in both splits (leakage check).
    """
    df       = pd.read_csv(csv_path)
    patients = df["Patient_Id"].unique()

    if len(patients) < 5:
        raise ValueError(
            f"Only {len(patients)} unique patients found. Need ≥ 5."
        )

    train_ids = patients[:18]
    val_ids   = patients[18:]

    overlap = set(train_ids) & set(val_ids)
    assert not overlap, f"Data leakage detected — patients in both splits: {overlap}"

    train_df = df[df["Patient_Id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["Patient_Id"].isin(val_ids)].reset_index(drop=True)

    logger.info(
        "Patient split — train: %d patients, val: %d patients",
        len(train_ids), len(val_ids),
    )

    return (
        PPGFusionDataset(dataframe=train_df, window_size=window_size),
        PPGFusionDataset(dataframe=val_df,   window_size=window_size),
    )


def build_dataloader(
    dataset: PPGFusionDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with settings safe for macOS MPS (num_workers=0)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # >0 causes multiprocessing issues on macOS MPS
        pin_memory=False,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
    )

    csv = Path(__file__).parent / "Archive_Dataset" / "clean-dataset.csv"
    if not csv.exists():
        sys.exit(f"ERROR: CSV not found at {csv}")

    print(f"\nLoading: {csv}")
    train_ds, val_ds = build_train_val_datasets(str(csv))
    print(f"Train → {train_ds}")
    print(f"Val   → {val_ds}")

    loader = build_dataloader(train_ds, batch_size=4)
    batch  = next(iter(loader))

    print(f"\nBatch shapes:")
    print(f"  1d_signal      : {batch['1d_signal'].shape}")       # [4, 1, 500]
    print(f"  2d_spectrogram : {batch['2d_spectrogram'].shape}")  # [4, 1, 64, 64]
    print(f"  3d_phase       : {batch['3d_phase'].shape}")        # [4, 3, 256]
    print(f"  label          : {batch['label'].shape}")           # [4, 1]
    print("\nfusion_data_pipeline.py  OK")
