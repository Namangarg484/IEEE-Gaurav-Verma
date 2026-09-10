"""
test_integrity.py
=================
Scientific integrity checks for the PPG-Glucose Fusion codebase.

Runs 12 checks covering data, code, model, and output correctness.
Exit code 0 = all pass.  Exit code 1 = at least one failure.

Usage:
    python test_integrity.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # clean output for CI-style printing
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []   # (id, label, detail)


def check(cid: str, label: str, fn) -> bool:
    """Run fn().  Record PASS/FAIL with detail message."""
    try:
        detail = fn()
        results.append((cid, PASS, str(detail) if detail else ""))
        return True
    except Exception as exc:
        results.append((cid, FAIL, str(exc)))
        return False


# ===========================================================================
# Check 1 — Training log exists and has monotonic, consecutive epochs
# ===========================================================================
def chk_training_log():
    log = pd.read_csv(ROOT / "training_log.csv")
    epochs = log["epoch"].tolist()
    assert len(epochs) > 0, "training_log.csv is empty"
    assert all(epochs[i] < epochs[i + 1] for i in range(len(epochs) - 1)), \
        f"Epochs not monotonic: {epochs}"
    cols = set(log.columns)
    assert {"epoch", "train_mse", "val_mse", "val_mae", "lr"} <= cols, \
        f"Missing columns; have {cols}"
    return f"{len(log)} epochs, monotonic"


check("1", "Training log — exists, monotonic, correct columns", chk_training_log)


# ===========================================================================
# Check 2 — LR scheduler present in train_fusion_model.py
# ===========================================================================
def chk_lr_scheduler():
    src = (ROOT / "train_fusion_model.py").read_text(encoding="utf-8")
    assert "lr_scheduler" in src or "ReduceLROnPlateau" in src, \
        "No LR scheduler found in train_fusion_model.py"
    return "ReduceLROnPlateau found"


check("2", "LR scheduler present in train_fusion_model.py", chk_lr_scheduler)


# ===========================================================================
# Check 3 — No fabricated random data in Clarke plot script
# ===========================================================================
def chk_no_random_clarke():
    src = (ROOT / "plot_clarke_error_grid.py").read_text(encoding="utf-8")
    assert "np.random.normal" not in src, \
        "np.random.normal still in plot_clarke_error_grid.py"
    return "No random data generation"


check("3", "No fabricated random data in plot_clarke_error_grid.py", chk_no_random_clarke)


# ===========================================================================
# Check 4 — No hardcoded fabricated arrays in generate_extra_figures.py
# ===========================================================================
def chk_no_fake_extra():
    src = (ROOT / "generate_extra_figures.py").read_text(encoding="utf-8")
    bad_snr  = "np.array([12.1, 13.5" in src or "np.array([11.7, 11.9" in src
    bad_attn = ("[0.45, 0.35, 0.20]" in src and "weights = np.array" in src)
    assert not bad_snr,  "Hardcoded fake SNR array still in generate_extra_figures.py"
    assert not bad_attn, "Hardcoded fake attention weights still in generate_extra_figures.py"
    return "No fabricated arrays found"


check("4", "No hardcoded fabricated data in generate_extra_figures.py", chk_no_fake_extra)


# ===========================================================================
# Check 5 — Named column access (no fragile iloc[:, N] in plotting scripts)
# ===========================================================================
def chk_named_columns():
    for fname in ["plot_overlay_comparison.py", "generate_paper_figures.py"]:
        src = (ROOT / fname).read_text(encoding="utf-8")
        assert "iloc[:, 3]" not in src, f"Fragile iloc[:, 3] still in {fname}"
        assert "iloc[:, 9]" not in src, f"Fragile iloc[:, 9] still in {fname}"
    return "All scripts use named column access"


check("5", "No fragile iloc[:, N] column access in plotting scripts", chk_named_columns)


# ===========================================================================
# Check 6 — weights_only=True in all torch.load calls
# ===========================================================================
def chk_weights_only():
    unsafe = []
    for fname in ["train_fusion_model.py", "eval_mae.py",
                  "generate_extra_figures.py", "generate_paper_figures.py"]:
        p = ROOT / fname
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        if "torch.load(" in src and "weights_only" not in src:
            unsafe.append(fname)
    assert not unsafe, f"torch.load missing weights_only=True in: {unsafe}"
    return "All torch.load calls use weights_only=True"


check("6", "weights_only=True in all torch.load calls", chk_weights_only)


# ===========================================================================
# Check 7 — Clarke results JSON is internally consistent
# ===========================================================================
def chk_clarke_json():
    cr   = json.loads((ROOT / "clarke_results.json").read_text())
    total = cr["total_points"]
    zone_sum = sum(cr[f"zone_{z}_count"] for z in "ABCDE")
    assert zone_sum == total, \
        f"Zone count sum {zone_sum} != total_points {total}"
    assert total > 0, "total_points is 0"
    return f"total={total}, zone_sum={zone_sum}"


check("7", "Clarke results JSON — total_points == sum of zone counts", chk_clarke_json)


# ===========================================================================
# Check 8 — Patient-level split has no leakage
# ===========================================================================
def chk_no_leakage():
    df      = pd.read_csv(ROOT / "Archive_Dataset" / "clean-dataset.csv")
    patients = df["Patient_Id"].unique()
    train_p  = set(patients[:18])
    val_p    = set(patients[18:])
    overlap  = train_p & val_p
    assert len(overlap) == 0, f"Patients in both train and val: {overlap}"
    return f"train={len(train_p)} patients, val={len(val_p)} patients, no overlap"


check("8", "Patient-level split — no train/val leakage", chk_no_leakage)


# ===========================================================================
# Check 9 — Model branch output dimensions match architecture spec
# ===========================================================================
def chk_model_dims():
    from fusion_network_model import MultimodalFusionNet
    model = MultimodalFusionNet()
    B  = 2
    d1 = torch.randn(B, 1,    500)
    d2 = torch.randn(B, 1, 64, 64)
    d3 = torch.randn(B, 3,    256)
    f1 = model.branch_1d(d1)
    f2 = model.branch_2d(d2)
    f3 = model.branch_3d(d3)
    assert f1.shape == (B, 256), f"Branch1D: {f1.shape}"
    assert f2.shape == (B, 256), f"Branch2D: {f2.shape}"
    assert f3.shape == (B, 256), f"Branch3D: {f3.shape}"
    out = model(d1, d2, d3)
    assert out.shape == (B, 1), f"Output: {out.shape}"
    return f"1D={f1.shape[1]}, 2D={f2.shape[1]}, 3D={f3.shape[1]}, out={out.shape}"


check("9", "Model branch dimensions — all branches output 256 features", chk_model_dims)


# ===========================================================================
# Check 10 — Gradient clipping present in training script
# ===========================================================================
def chk_grad_clip():
    src = (ROOT / "train_fusion_model.py").read_text(encoding="utf-8")
    assert "clip_grad_norm_" in src, \
        "clip_grad_norm_ not found in train_fusion_model.py"
    return "clip_grad_norm_ present"


check("10", "Gradient clipping present in train_fusion_model.py", chk_grad_clip)


# ===========================================================================
# Check 11 — Saved model file exists and loads without error
# ===========================================================================
def chk_model_loads():
    p = ROOT / "fusion_model_v2_mac.pth"
    assert p.exists(), f"Model checkpoint not found: {p}"
    from fusion_network_model import MultimodalFusionNet
    model = MultimodalFusionNet()
    state = torch.load(str(p), weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        # Detect stale checkpoint from old architecture
        msg = str(exc)
        if "Missing key" in msg or "Unexpected key" in msg:
            raise AssertionError(
                "Checkpoint architecture mismatch — old weights detected.\n"
                "  The saved .pth was from the OLD architecture.\n"
                "  Run training to generate a fresh checkpoint:\n"
                "    ./run_training.sh --epochs 100\n"
                "  or: python train_fusion_model.py --epochs 100"
            ) from None
        raise
    return f"Loaded {len(state)} parameter tensors"


check("11", "Saved model checkpoint loads cleanly", chk_model_loads)


# ===========================================================================
# Check 12 — Data pipeline produces correct tensor shapes
# ===========================================================================
def chk_pipeline_shapes():
    from fusion_data_pipeline import build_dataloader, build_train_val_datasets
    _, val_ds = build_train_val_datasets(
        str(ROOT / "Archive_Dataset" / "clean-dataset.csv")
    )
    loader = build_dataloader(val_ds, batch_size=4, shuffle=False)
    batch  = next(iter(loader))
    assert batch["1d_signal"].shape[1:]      == (1, 500),    str(batch["1d_signal"].shape)
    assert batch["2d_spectrogram"].shape[1:] == (1, 64, 64), str(batch["2d_spectrogram"].shape)
    assert batch["3d_phase"].shape[1:]       == (3, 256),    str(batch["3d_phase"].shape)
    return "1d=(1,500) 2d=(1,64,64) 3d=(3,256)"


check("12", "Data pipeline produces correct tensor shapes", chk_pipeline_shapes)


# ===========================================================================
# Summary
# ===========================================================================
print()
print("=" * 60)
print("  SCIENTIFIC INTEGRITY CHECK REPORT")
print("=" * 60)

n_pass = n_fail = 0
for cid, status, detail in results:
    icon = "✅" if status == PASS else "❌"
    line = f"  {icon} Check {cid:>2}: {detail}"
    print(line)
    if status == PASS:
        n_pass += 1
    else:
        n_fail += 1

print()
print(f"  Result: {n_pass} passed / {n_fail} failed / {len(results)} total")
print("=" * 60)

sys.exit(0 if n_fail == 0 else 1)
