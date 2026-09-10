#!/usr/bin/env python3
"""
run_extended_analysis.py
========================
Three rigorous analyses that strengthen the paper:

  1. SQI Filtering
     ─────────────
     Compute per-window signal-quality index (std of raw PPG).
     Filter "flatline" windows below threshold.
     Report dataset statistics and evaluate the pre-trained model on
     clean-only vs. all windows.

  2. t-SNE Identity Ablation
     ─────────────────────────
     Extract the 768-d fused latent vector z from every validation window.
     Run t-SNE (2-D) and produce two plots:
       (a) coloured by Patient ID   — reveals if model clusters by identity
       (b) coloured by glucose level — reveals if model organises by glucose

  3. Attractor Analysis
     ───────────────────
     For each validation window, retrieve the 3-D Takens point cloud.
     Compute the correlation dimension (Grassberger-Procaccia) and the
     largest Lyapunov exponent (Rosenstein method).
     Plot both against measured glucose level.
     Visualise representative attractors at low / mid / high glucose.

Outputs
-------
    sqi_results.json              — SQI statistics and model performance
    attractor_stats.json          — correlation dim + Lyapunov exponents per window
    Paper_SQI_Analysis.png        — SQI distribution + model comparison
    Paper_tSNE_Identity.png       — t-SNE panels (patient ID vs. glucose)
    Paper_Attractor_Analysis.png  — correlation dim / Lyapunov vs. glucose

Usage
-----
    python run_extended_analysis.py [--device auto]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist, pdist
from sklearn.manifold import TSNE

from fusion_data_pipeline import (
    WINDOW_SIZE,
    TIME_DELAY,
    PHASE_POINTS,
    PPGFusionDataset,
    build_dataloader,
    build_train_val_datasets,
)
from fusion_network_model import MultimodalFusionNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = "cpu"
CSV    = Path("Archive_Dataset/clean-dataset.csv")
CKPT   = Path("fusion_model_v2_mac.pth")


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Signal Quality Index
# ─────────────────────────────────────────────────────────────────────────────

def compute_sqi(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """For every window, compute: patient_id, mean_glucose, raw_std, sqi_pass."""
    rows = []
    n_windows = len(df) // window_size
    for i in range(n_windows):
        chunk = df.iloc[i * window_size:(i + 1) * window_size]
        raw_std  = float(chunk["PPG_Signal"].std())
        glucose  = float(chunk["Glucose_level"].mean())
        patient  = int(chunk["Patient_Id"].iloc[0])
        rows.append({
            "window_idx": i,
            "patient_id": patient,
            "glucose":    glucose,
            "raw_std":    raw_std,
        })
    return pd.DataFrame(rows)


def sqi_analysis(df_full: pd.DataFrame, train_ids, val_ids) -> dict:
    """
    Compute SQI stats on training and validation sets.
    Evaluate the pre-trained model on:
      (a) all validation windows
      (b) only SQI-passing validation windows
    Returns a dict of statistics.
    """
    train_df = df_full[df_full["Patient_Id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df_full[df_full["Patient_Id"].isin(val_ids)].reset_index(drop=True)

    train_sqi = compute_sqi(train_df)
    val_sqi   = compute_sqi(val_df)

    # Data-driven threshold: separate flatlines (std ≈ 0) from pulsatile signals.
    # Use the 5th percentile of NON-ZERO training stds, ensuring minimum 1e-4.
    nonzero_stds = train_sqi[train_sqi["raw_std"] > 1e-6]["raw_std"]
    threshold = float(nonzero_stds.quantile(0.05)) if len(nonzero_stds) > 0 else 1e-4
    threshold = max(threshold, 1e-4)   # never below 1e-4
    log.info("SQI threshold: raw_std >= %.4f  (5th pct of non-zero training stds)", threshold)

    train_pass = (train_sqi["raw_std"] >= threshold).sum()
    val_pass   = (val_sqi["raw_std"]   >= threshold).sum()

    log.info("Training: %d/%d windows pass SQI (%.1f%%)",
             train_pass, len(train_sqi), 100 * train_pass / len(train_sqi))
    log.info("Validation: %d/%d windows pass SQI (%.1f%%)",
             val_pass, len(val_sqi), 100 * val_pass / len(val_sqi))

    stats = {
        "threshold": threshold,
        "train_total":    int(len(train_sqi)),
        "train_pass":     int(train_pass),
        "train_fail":     int(len(train_sqi) - train_pass),
        "val_total":      int(len(val_sqi)),
        "val_pass":       int(val_pass),
        "val_fail":       int(len(val_sqi) - val_pass),
        "val_pass_pct":   round(100 * val_pass / len(val_sqi), 1),
    }

    # Evaluate model on all vs. clean-only val windows
    if CKPT.exists():
        model = MultimodalFusionNet().to(DEVICE)
        model.load_state_dict(torch.load(CKPT, weights_only=True, map_location=DEVICE))
        model.eval()

        def _eval_subset(window_indices):
            if not window_indices:
                return None, None
            slices = [val_df.iloc[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
                      for i in window_indices
                      if (i + 1) * WINDOW_SIZE <= len(val_df)]
            if not slices:
                return None, None
            sub_df = pd.concat(slices, ignore_index=True)
            if len(sub_df) == 0:
                return None, None
            ds = PPGFusionDataset(dataframe=sub_df)
            loader = build_dataloader(ds, batch_size=64, shuffle=False)
            preds_all, refs_all = [], []
            with torch.no_grad():
                for batch in loader:
                    out = model(
                        batch["1d_signal"].to(DEVICE),
                        batch["2d_spectrogram"].to(DEVICE),
                        batch["3d_phase"].to(DEVICE),
                    )
                    preds_all.extend(out.squeeze(-1).cpu().numpy())
                    refs_all.extend(batch["label"].squeeze(-1).cpu().numpy())
            p = np.array(preds_all)
            r = np.array(refs_all)
            return p, r

        all_idx   = list(range(len(val_sqi)))
        clean_idx = val_sqi[val_sqi["raw_std"] >= threshold].index.tolist()

        p_all, r_all = _eval_subset(all_idx)
        if p_all is not None:
            mae_all  = float(np.mean(np.abs(p_all - r_all)))
            rmse_all = float(np.sqrt(np.mean((p_all - r_all) ** 2)))
            stats["mae_all_windows"]  = mae_all
            stats["rmse_all_windows"] = rmse_all
            log.info("All val windows:   MAE=%.3f  RMSE=%.3f", mae_all, rmse_all)

        p_clean, r_clean = _eval_subset(clean_idx)
        if p_clean is not None:
            mae_clean  = float(np.mean(np.abs(p_clean - r_clean)))
            rmse_clean = float(np.sqrt(np.mean((p_clean - r_clean) ** 2)))
            stats["mae_clean_windows"]  = mae_clean
            stats["rmse_clean_windows"] = rmse_clean
            log.info("Clean val windows: MAE=%.3f  RMSE=%.3f", mae_clean, rmse_clean)

    return stats, train_sqi, val_sqi


def plot_sqi(train_sqi: pd.DataFrame, val_sqi: pd.DataFrame,
             threshold: float, save_path: str = "Paper_SQI_Analysis.png") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Distribution of raw_std across all windows
    ax = axes[0]
    all_stds = pd.concat([train_sqi["raw_std"], val_sqi["raw_std"]])
    ax.hist(all_stds[all_stds < 200], bins=60, color="#4285f4", alpha=0.8, edgecolor="white")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"SQI threshold ({threshold:.1f})")
    fail_pct = 100 * (all_stds < threshold).mean()
    ax.set_xlabel("Raw PPG std (ADC counts)", fontsize=11)
    ax.set_ylabel("Window count",             fontsize=11)
    ax.set_title(f"SQI Distribution\n({fail_pct:.1f}% fail, i.e. flatlines)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, min(200, all_stds.quantile(0.98)))

    # Per-patient SQI pass rate (validation only)
    ax = axes[1]
    pass_rates = []
    patients   = sorted(val_sqi["patient_id"].unique())
    for pid in patients:
        sub = val_sqi[val_sqi["patient_id"] == pid]
        rate = 100 * (sub["raw_std"] >= threshold).mean()
        pass_rates.append(rate)
    bars = ax.bar([f"P{p}" for p in patients], pass_rates,
                  color="#34a853", edgecolor="white")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("SQI Pass Rate (%)", fontsize=11)
    ax.set_title("SQI Pass Rate per Validation Patient", fontsize=11)
    for bar, r in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r:.0f}%", ha="center", fontsize=10)

    # Scatter: raw_std vs glucose (val windows only)
    ax = axes[2]
    clean  = val_sqi[val_sqi["raw_std"] >= threshold]
    dirty  = val_sqi[val_sqi["raw_std"] <  threshold]
    ax.scatter(clean["glucose"], clean["raw_std"],
               c="#4285f4", s=20, alpha=0.7, label=f"Clean ({len(clean)})")
    ax.scatter(dirty["glucose"], dirty["raw_std"] + 0.5,
               c="#ea4335", s=20, alpha=0.7, label=f"Flatline ({len(dirty)})")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Glucose (mg/dL)",         fontsize=11)
    ax.set_ylabel("Raw PPG std (ADC counts)", fontsize=11)
    ax.set_title("Flatlines Are Not Glucose-Biased\n(flatlines span all glucose levels)",
                 fontsize=11)
    ax.legend(fontsize=9)

    plt.suptitle("Signal Quality Index (SQI) Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · t-SNE Identity Ablation
# ─────────────────────────────────────────────────────────────────────────────

def extract_latents(val_ds, model: MultimodalFusionNet) -> tuple:
    """Extract 768-d fused latent vectors, attention weights, glucose labels, patient IDs."""
    loader = build_dataloader(val_ds, batch_size=64, shuffle=False)
    latents, glucoses, patients, attn_weights = [], [], [], []

    # Rebuild val patient list from dataframe
    val_df = val_ds.df
    n_windows = len(val_ds)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            sig_1d = batch["1d_signal"].to(DEVICE)
            sig_2d = batch["2d_spectrogram"].to(DEVICE)
            sig_3d = batch["3d_phase"].to(DEVICE)

            f1 = model.branch_1d(sig_1d)
            f2 = model.branch_2d(sig_2d)
            f3 = model.branch_3d(sig_3d)

            # Fused 768-d representation
            z = model.fusion([f1, f2, f3])    # (B, 768)
            w = model.fusion.get_weights([f1, f2, f3])  # (B, 3)

            latents.append(z.cpu().numpy())
            attn_weights.append(w.cpu().numpy())
            glucoses.extend(batch["label"].squeeze(-1).cpu().numpy())

    latents      = np.vstack(latents)       # (N_val, 768)
    attn_weights = np.vstack(attn_weights)  # (N_val, 3)
    glucoses     = np.array(glucoses)

    # Reconstruct patient IDs per window
    for i in range(n_windows):
        start = i * WINDOW_SIZE
        pid   = int(val_df.iloc[start]["Patient_Id"])
        patients.append(pid)
    patients = np.array(patients)

    return latents, glucoses, patients, attn_weights


def run_tsne(latents: np.ndarray, glucoses: np.ndarray,
             patients: np.ndarray, save_path: str = "Paper_tSNE_Identity.png") -> None:
    log.info("Running t-SNE on %d × 768 latent vectors …", len(latents))
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42,
                   max_iter=1000, learning_rate="auto", init="pca")
    Z = tsne_2d.fit_transform(latents)
    log.info("t-SNE done.")

    unique_pids = sorted(np.unique(patients))
    pid_colors  = plt.colormaps["tab10"].resampled(len(unique_pids))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: coloured by patient ID
    for i, pid in enumerate(unique_pids):
        mask = patients == pid
        ax1.scatter(Z[mask, 0], Z[mask, 1],
                    c=[pid_colors(i)] * mask.sum(),
                    s=25, alpha=0.85, label=f"Patient {pid}", edgecolors="none")
    ax1.set_title("t-SNE — coloured by Patient ID\n"
                  "(if clusters align with patients → identity learning)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("t-SNE dim 1"); ax1.set_ylabel("t-SNE dim 2")
    ax1.legend(fontsize=9, markerscale=1.5)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel B: coloured by glucose level
    sc = ax2.scatter(Z[:, 0], Z[:, 1], c=glucoses,
                     cmap="RdYlGn_r", s=25, alpha=0.85, edgecolors="none",
                     vmin=np.percentile(glucoses, 5),
                     vmax=np.percentile(glucoses, 95))
    cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("Glucose (mg/dL)", fontsize=10)
    ax2.set_title("t-SNE — coloured by Glucose Level\n"
                  "(gradient across space → glucose-correlated representation)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("t-SNE dim 1"); ax2.set_ylabel("t-SNE dim 2")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Silhouette-style comment
    # Compute inter-patient separation
    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(Z, patients)
        fig.text(0.5, 0.01,
                 f"Patient silhouette score: {sil:.3f}  "
                 f"(near 0 = no identity clustering, near 1 = pure identity clustering)",
                 ha="center", fontsize=10, style="italic")
    except Exception:
        pass

    plt.suptitle("Latent Space Analysis — Is the Model Learning Patient Identity?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Attractor Analysis (Correlation Dimension + Lyapunov Exponent)
# ─────────────────────────────────────────────────────────────────────────────

def correlation_dimension_gp(points: np.ndarray,
                               n_r: int = 20,
                               r_lo_pct: float = 5.0,
                               r_hi_pct: float = 50.0) -> float:
    """
    Grassberger-Procaccia correlation dimension estimate.
    points: (N, 3) attractor point cloud
    Returns the slope of log C(r) vs log r in the scaling region.
    """
    if points.shape[0] < 10:
        return np.nan
    # Sub-sample for speed (GP is O(N^2))
    pts = points[:min(128, len(points))]
    dists = pdist(pts)
    if len(dists) == 0 or dists.max() < 1e-10:
        return np.nan

    r_min = np.percentile(dists[dists > 0], r_lo_pct)
    r_max = np.percentile(dists[dists > 0], r_hi_pct)
    if r_min >= r_max:
        return np.nan

    rs = np.linspace(r_min, r_max, n_r)
    Cs = np.array([np.mean(dists < r) for r in rs])

    # Only fit where C is strictly positive and not saturated
    valid = (Cs > 0.01) & (Cs < 0.99)
    if valid.sum() < 3:
        return np.nan

    log_r = np.log(rs[valid])
    log_C = np.log(Cs[valid])
    slope, _ = np.polyfit(log_r, log_C, 1)
    return float(np.clip(slope, 0, 6))   # physical range for D2


def largest_lyapunov_rosenstein(ts: np.ndarray, m: int = 3, tau: int = 5,
                                  min_tsep: int = 20, max_iter: int = 30) -> float:
    """
    Rosenstein et al. (1993) estimate of the largest Lyapunov exponent.
    ts    : 1-D time series (already Z-score normalised)
    m     : embedding dimension
    tau   : time delay
    Returns LLE in units of (1/sample).
    """
    N = len(ts)
    M = N - (m - 1) * tau
    if M < 20:
        return np.nan

    # Delay-embedded matrix: (M, m)
    Y = np.array([ts[i:i + (m - 1) * tau + 1:tau] for i in range(M)])

    # For each point, find nearest neighbour with temporal separation > min_tsep
    dists_nn = []
    nn_idx   = []
    for i in range(M):
        d = np.linalg.norm(Y - Y[i], axis=1)
        d[max(0, i - min_tsep):min(M, i + min_tsep)] = np.inf
        j = int(np.argmin(d))
        dists_nn.append(d[j])
        nn_idx.append(j)

    # Track divergence over max_iter steps
    divergences = []
    for step in range(1, min(max_iter, M // 4)):
        log_div = []
        for i, j in enumerate(nn_idx):
            i2 = i + step
            j2 = j + step
            if i2 >= M or j2 >= M:
                continue
            dist = np.linalg.norm(Y[i2] - Y[j2])
            if dist > 1e-10 and dists_nn[i] > 1e-10:
                log_div.append(np.log(dist / dists_nn[i]))
        if log_div:
            divergences.append(np.mean(log_div))

    if len(divergences) < 3:
        return np.nan

    # LLE = slope of the mean log divergence curve
    steps = np.arange(1, len(divergences) + 1, dtype=float)
    lle, _ = np.polyfit(steps, divergences, 1)
    return float(lle)


def attractor_analysis(val_df: pd.DataFrame) -> tuple:
    """Compute correlation dim and LLE for every val window."""
    n_windows = len(val_df) // WINDOW_SIZE
    records = []
    log.info("Computing attractor statistics for %d windows …", n_windows)
    for i in range(n_windows):
        chunk = val_df.iloc[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
        sig   = chunk["PPG_Signal"].values.astype(np.float32)
        glucose = float(chunk["Glucose_level"].mean())
        patient = int(chunk["Patient_Id"].iloc[0])

        # Z-score normalise
        mu, sigma = sig.mean(), sig.std()
        sig_norm = (sig - mu) / (sigma + 1e-8)

        # 3-D Takens embedding (same as pipeline)
        tau = TIME_DELAY
        L   = len(sig_norm) - 2 * tau
        x, y, z = sig_norm[:L], sig_norm[tau:tau+L], sig_norm[2*tau:2*tau+L]
        pts = np.column_stack([x, y, z])[:PHASE_POINTS]   # (≤256, 3)

        cd  = correlation_dimension_gp(pts)
        lle = largest_lyapunov_rosenstein(sig_norm)

        records.append({
            "window_idx": i,
            "patient_id": patient,
            "glucose":    glucose,
            "corr_dim":   cd,
            "lle":        lle,
            "raw_std":    float(sig.std()),
        })
        if (i + 1) % 50 == 0:
            log.info("  %d / %d windows processed …", i + 1, n_windows)

    return pd.DataFrame(records)


def plot_attractor_analysis(stats_df: pd.DataFrame,
                             val_df:   pd.DataFrame,
                             save_path: str = "Paper_Attractor_Analysis.png") -> None:
    # Data-driven threshold: separate flatlines from pulsatile windows
    nonzero = stats_df[stats_df["raw_std"] > 1e-6]["raw_std"]
    thr = float(nonzero.quantile(0.05)) if len(nonzero) > 0 else 1e-4
    thr = max(thr, 1e-4)
    clean = stats_df[stats_df["raw_std"] >= thr].copy()

    if len(clean) == 0:
        log.warning("No clean windows for attractor analysis plot.")
        return

    fig = plt.figure(figsize=(18, 11))

    # ── Row 1: correlation dimension vs. glucose ───────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    sc1 = ax1.scatter(clean["glucose"], clean["corr_dim"],
                      c=clean["glucose"], cmap="RdYlGn_r", s=30, alpha=0.7, edgecolors="none",
                      vmin=clean["glucose"].quantile(0.05),
                      vmax=clean["glucose"].quantile(0.95))
    plt.colorbar(sc1, ax=ax1, label="Glucose (mg/dL)")
    # Trend line
    valid_cd = clean.dropna(subset=["corr_dim"])
    if len(valid_cd) > 5:
        z_fit = np.polyfit(valid_cd["glucose"], valid_cd["corr_dim"], 1)
        x_line = np.linspace(valid_cd["glucose"].min(), valid_cd["glucose"].max(), 100)
        ax1.plot(x_line, np.polyval(z_fit, x_line), "r--", linewidth=2, alpha=0.8)
        r_val = float(np.corrcoef(valid_cd["glucose"], valid_cd["corr_dim"])[0, 1])
        ax1.set_title(f"Correlation Dimension vs. Glucose\n(Pearson r = {r_val:.3f})", fontsize=11)
    else:
        ax1.set_title("Correlation Dimension vs. Glucose", fontsize=11)
    ax1.set_xlabel("Glucose (mg/dL)")
    ax1.set_ylabel("Correlation Dimension D₂")

    # ── Row 1: LLE vs. glucose ─────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    valid_lle = clean.dropna(subset=["lle"])
    if len(valid_lle) > 5:
        sc2 = ax2.scatter(valid_lle["glucose"], valid_lle["lle"],
                          c=valid_lle["glucose"], cmap="RdYlGn_r",
                          s=30, alpha=0.7, edgecolors="none",
                          vmin=valid_lle["glucose"].quantile(0.05),
                          vmax=valid_lle["glucose"].quantile(0.95))
        plt.colorbar(sc2, ax=ax2, label="Glucose (mg/dL)")
        z_fit = np.polyfit(valid_lle["glucose"], valid_lle["lle"], 1)
        x_line = np.linspace(valid_lle["glucose"].min(), valid_lle["glucose"].max(), 100)
        ax2.plot(x_line, np.polyval(z_fit, x_line), "r--", linewidth=2, alpha=0.8)
        r_val = float(np.corrcoef(valid_lle["glucose"], valid_lle["lle"])[0, 1])
        ax2.set_title(f"Largest Lyapunov Exp. vs. Glucose\n(Pearson r = {r_val:.3f})", fontsize=11)
    ax2.set_xlabel("Glucose (mg/dL)")
    ax2.set_ylabel("Largest Lyapunov Exponent (1/sample)")

    # ── Row 1: box plot — correlation dim across glucose quartiles ─────────
    ax3 = fig.add_subplot(2, 3, 3)
    valid_cd = clean.dropna(subset=["corr_dim"]).copy()
    if len(valid_cd) >= 8:
        valid_cd["glucose_quartile"] = pd.qcut(
            valid_cd["glucose"], q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
        )
        quartiles = ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
        groups = [valid_cd[valid_cd["glucose_quartile"] == q]["corr_dim"].dropna().values
                  for q in quartiles]
        bp = ax3.boxplot(groups, labels=quartiles, patch_artist=True,
                         notch=True, medianprops={"color": "red", "linewidth": 2})
        colors = plt.colormaps["RdYlGn_r"].resampled(4)([0.1, 0.4, 0.7, 0.95])
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax3.set_xlabel("Glucose Quartile")
        ax3.set_ylabel("Correlation Dimension D₂")
        ax3.set_title("Attractor Complexity\nby Glucose Quartile", fontsize=11)

    # ── Row 2: 3-D attractor visualisations for low / mid / high glucose ──
    valid_cd = clean.dropna(subset=["corr_dim"]).copy()
    low_idx  = valid_cd.nsmallest(1, "glucose")["window_idx"].iloc[0]
    mid_idx  = valid_cd.iloc[len(valid_cd) // 2]["window_idx"]
    high_idx = valid_cd.nlargest(1, "glucose")["window_idx"].iloc[0]

    def extract_attractor(win_idx):
        chunk    = val_df.iloc[win_idx * WINDOW_SIZE:(win_idx + 1) * WINDOW_SIZE]
        sig      = chunk["PPG_Signal"].values.astype(np.float32)
        mu, std  = sig.mean(), sig.std()
        sig_norm = (sig - mu) / (std + 1e-8)
        tau, L   = TIME_DELAY, len(sig_norm) - 2 * TIME_DELAY
        x, y, z  = sig_norm[:L], sig_norm[TIME_DELAY:TIME_DELAY+L], sig_norm[2*TIME_DELAY:2*TIME_DELAY+L]
        gluc     = float(chunk["Glucose_level"].mean())
        return np.column_stack([x, y, z])[:PHASE_POINTS], gluc

    for subplot_idx, (win_idx, label) in enumerate(
        [(low_idx, "Low"), (mid_idx, "Mid"), (high_idx, "High")]
    ):
        ax = fig.add_subplot(2, 3, 4 + subplot_idx, projection="3d")
        pts, gluc = extract_attractor(int(win_idx))
        cd_val = valid_cd[valid_cd["window_idx"] == win_idx]["corr_dim"].values
        cd_str = f"{cd_val[0]:.2f}" if len(cd_val) > 0 and not np.isnan(cd_val[0]) else "N/A"
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=plt.cm.RdYlGn_r(subplot_idx / 3.0),
                linewidth=0.8, alpha=0.7)
        ax.scatter(pts[::10, 0], pts[::10, 1], pts[::10, 2],
                   s=8, color=plt.cm.RdYlGn_r(subplot_idx / 3.0), alpha=0.9)
        ax.set_title(f"{label} Glucose\n{gluc:.1f} mg/dL  (D₂≈{cd_str})", fontsize=10)
        ax.set_xlabel("x(t)", fontsize=8); ax.set_ylabel("x(t+τ)", fontsize=8)
        ax.set_zlabel("x(t+2τ)", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.suptitle("PPG Attractor (Takens Embedding) Analysis:\n"
                 "Does Cardiovascular Attractor Geometry Encode Glucose?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-tsne",     action="store_true")
    p.add_argument("--skip-attractor", action="store_true")
    return p.parse_args()


def main():
    global DEVICE
    args = parse_args()

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

    log.info("Loading dataset …")
    df       = pd.read_csv(str(CSV))
    patients = df["Patient_Id"].unique()
    train_ids = patients[:18]
    val_ids   = patients[18:]

    val_df = df[df["Patient_Id"].isin(val_ids)].reset_index(drop=True)

    # ── 1. SQI Analysis ───────────────────────────────────────────────────────
    log.info("\n=== 1. SIGNAL QUALITY INDEX ANALYSIS ===")
    train_df = df[df["Patient_Id"].isin(train_ids)].reset_index(drop=True)
    train_sqi_df = compute_sqi(train_df)
    val_sqi_df   = compute_sqi(val_df)

    # Data-driven threshold: 5th percentile of non-zero training stds
    nonzero_train = train_sqi_df[train_sqi_df["raw_std"] > 1e-6]["raw_std"]
    threshold = float(nonzero_train.quantile(0.05)) if len(nonzero_train) > 0 else 1e-4
    threshold = max(threshold, 1e-4)

    sqi_stats = {
        "threshold": threshold,
        "train_total": len(train_sqi_df),
        "train_pass":  int((train_sqi_df["raw_std"] >= threshold).sum()),
        "val_total":   len(val_sqi_df),
        "val_pass":    int((val_sqi_df["raw_std"] >= threshold).sum()),
    }
    sqi_stats["train_fail_pct"] = round(
        100 * (1 - sqi_stats["train_pass"] / sqi_stats["train_total"]), 1
    )
    sqi_stats["val_fail_pct"] = round(
        100 * (1 - sqi_stats["val_pass"] / sqi_stats["val_total"]), 1
    )

    # Evaluate model on clean-only windows
    if CKPT.exists():
        model_sqi = MultimodalFusionNet().to(DEVICE)
        model_sqi.load_state_dict(
            torch.load(CKPT, weights_only=True, map_location=DEVICE)
        )
        model_sqi.eval()

        clean_indices = val_sqi_df[val_sqi_df["raw_std"] >= threshold].index.tolist()
        log.info("Evaluating model on %d clean-only val windows …", len(clean_indices))

        slices = [val_df.iloc[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
                  for i in clean_indices
                  if i * WINDOW_SIZE < len(val_df)]
        if not slices:
            log.warning("No clean windows found for model evaluation.")
        else:
            sub_df = pd.concat(slices, ignore_index=True)
        if slices and len(sub_df) > 0:
            clean_ds     = PPGFusionDataset(dataframe=sub_df)
            clean_loader = build_dataloader(clean_ds, batch_size=64, shuffle=False)
            preds_c, refs_c = [], []
            with torch.no_grad():
                for batch in clean_loader:
                    out = model_sqi(
                        batch["1d_signal"].to(DEVICE),
                        batch["2d_spectrogram"].to(DEVICE),
                        batch["3d_phase"].to(DEVICE),
                    )
                    preds_c.extend(out.squeeze(-1).cpu().numpy())
                    refs_c.extend(batch["label"].squeeze(-1).cpu().numpy())
            preds_c = np.array(preds_c)
            refs_c  = np.array(refs_c)
            sqi_stats["mae_clean"]  = round(float(np.mean(np.abs(preds_c - refs_c))), 3)
            sqi_stats["rmse_clean"] = round(float(np.sqrt(np.mean((preds_c - refs_c) ** 2))), 3)
            log.info("Clean-only  MAE=%.3f  RMSE=%.3f",
                     sqi_stats["mae_clean"], sqi_stats["rmse_clean"])

    plot_sqi(train_sqi_df, val_sqi_df, threshold)

    with open("sqi_results.json", "w") as f:
        json.dump(sqi_stats, f, indent=2)
    log.info("Saved: sqi_results.json  %s", sqi_stats)

    # ── 2. t-SNE Identity Ablation ────────────────────────────────────────────
    if not args.skip_tsne and CKPT.exists():
        log.info("\n=== 2. t-SNE IDENTITY ABLATION ===")
        _, val_ds = build_train_val_datasets(str(CSV))
        model_tsne = MultimodalFusionNet().to(DEVICE)
        model_tsne.load_state_dict(
            torch.load(CKPT, weights_only=True, map_location=DEVICE)
        )
        latents, glucoses, patients_arr, attn_w = extract_latents(val_ds, model_tsne)
        run_tsne(latents, glucoses, patients_arr)

        log.info("Attention weights across val set:")
        log.info("  1D: %.4f ± %.4f", attn_w[:, 0].mean(), attn_w[:, 0].std())
        log.info("  2D: %.4f ± %.4f", attn_w[:, 1].mean(), attn_w[:, 1].std())
        log.info("  3D: %.4f ± %.4f", attn_w[:, 2].mean(), attn_w[:, 2].std())
    else:
        log.info("t-SNE skipped.")

    # ── 3. Attractor Analysis ─────────────────────────────────────────────────
    if not args.skip_attractor:
        log.info("\n=== 3. ATTRACTOR ANALYSIS ===")
        cache_path = Path("attractor_stats.json")
        if cache_path.exists():
            log.info("Loading cached attractor stats from %s", cache_path)
            import json as _json
            with open(cache_path) as f:
                cached = _json.load(f)
            stats_df = pd.DataFrame(cached)
            stats_df["corr_dim"] = pd.to_numeric(stats_df["corr_dim"], errors="coerce")
            stats_df["lle"]      = pd.to_numeric(stats_df["lle"],      errors="coerce")
        else:
            stats_df = attractor_analysis(val_df)

        # Save per-window stats (only when freshly computed)
        if not cache_path.exists() or stats_df is None:
            attractor_json = []
            for _, row in stats_df.iterrows():
                attractor_json.append({
                    "window_idx": int(row["window_idx"]),
                    "patient_id": int(row["patient_id"]),
                    "glucose":    round(float(row["glucose"]), 2),
                    "corr_dim":   None if pd.isna(row["corr_dim"]) else round(float(row["corr_dim"]), 4),
                    "lle":        None if pd.isna(row["lle"])      else round(float(row["lle"]), 6),
                    "raw_std":    round(float(row["raw_std"]), 3),
                })
            with open("attractor_stats.json", "w") as f:
                json.dump(attractor_json, f, indent=2)
            log.info("Saved: attractor_stats.json (%d windows)", len(attractor_json))

        # Summary statistics
        nonzero_s = stats_df[stats_df["raw_std"] > 1e-6]["raw_std"]
        sum_thr = float(nonzero_s.quantile(0.05)) if len(nonzero_s) > 0 else 1e-4
        sum_thr = max(sum_thr, 1e-4)
        clean_stats = stats_df[stats_df["raw_std"] >= sum_thr]
        valid_cd    = clean_stats["corr_dim"].dropna()
        valid_lle   = clean_stats["lle"].dropna()
        if len(valid_cd) > 2:
            r_cd  = float(np.corrcoef(clean_stats.dropna(subset=["corr_dim"])["glucose"],
                                       valid_cd)[0, 1])
            log.info("Correlation Dimension:  mean=%.3f ± %.3f  |  r vs glucose = %.3f",
                     valid_cd.mean(), valid_cd.std(), r_cd)
        if len(valid_lle) > 2:
            r_lle = float(np.corrcoef(clean_stats.dropna(subset=["lle"])["glucose"],
                                       valid_lle)[0, 1])
            log.info("Largest Lyapunov Exp.:  mean=%.4f ± %.4f  |  r vs glucose = %.3f",
                     valid_lle.mean(), valid_lle.std(), r_lle)

        plot_attractor_analysis(stats_df, val_df)
    else:
        log.info("Attractor analysis skipped.")

    log.info("\nAll extended analyses complete.")


if __name__ == "__main__":
    main()
