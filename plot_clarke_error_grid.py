"""
plot_clarke_error_grid.py
=========================
Clarke Error Grid Analysis — classification logic and plot generation.

Zone definitions (standard Clarke 2005)
----------------------------------------
    A  Clinically accurate (within 20% of reference, or within 20 mg/dL
       when reference < 70 mg/dL)
    B  Benign inaccuracy (would not lead to dangerous treatment decision)
    C  Overcorrection within euglycaemic range (potentially dangerous)
    D  Clinically dangerous silent errors (failure to detect hypo/hyper)
    E  Opposite clinical action (most dangerous — reversed hypo/hyper)

Correctness notes
-----------------
    Zone E is checked BEFORE Zone D to prevent misclassification at the
    boundary (low ref + high pred or vice versa).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

ZONE_ORDER = ("A", "B", "C", "D", "E")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _classify_point(ref: float, pred: float) -> str:
    """Classify a single (reference, prediction) pair into Clarke zone A–E."""
    # Zone A — clinically accurate
    if ref < 70:
        if abs(pred - ref) <= 20:
            return "A"
    else:
        if abs(pred - ref) <= 0.20 * ref:
            return "A"

    # Zone E — opposite clinical action (checked BEFORE D to avoid boundary overlap)
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"

    # Zone D — clinically dangerous silent error
    if (ref >= 240 and pred <= 180) or (ref <= 70 and 70 <= pred < 180):
        return "D"
    if 70 <= ref <= 180 and (pred <= 70 or pred >= 240):
        return "D"

    # Zone C — overcorrection in euglycaemic range
    if 70 < ref < 180 and pred > ref + 110:
        return "C"
    if 70 < ref < 180 and pred < ref - 110:
        return "C"

    # Zone B — everything else (benign inaccuracy)
    return "B"


def summarize_clarke_zones(
    refs: np.ndarray, preds: np.ndarray
) -> dict:
    """Count Clarke zones for paired arrays.  Returns a summary dict."""
    counts = {z: 0 for z in ZONE_ORDER}
    for r, p in zip(refs, preds):
        counts[_classify_point(float(r), float(p))] += 1

    n = len(refs)
    result = {"total_points": int(n)}
    for z in ZONE_ORDER:
        result[f"zone_{z}_count"] = int(counts[z])
        result[f"zone_{z}_pct"]   = float(counts[z] / n * 100) if n else 0.0
    return result


# ---------------------------------------------------------------------------
# Plot boundaries helper
# ---------------------------------------------------------------------------
def _draw_clarke_boundaries(ax, max_val: float) -> None:
    """Draw standard Clarke Error Grid boundary lines."""
    kw_g  = dict(color="green",  linewidth=1.2)
    kw_r  = dict(color="red",    linewidth=1.2, linestyle="--")
    kw_p  = dict(color="purple", linewidth=1.2, linestyle="-.")

    # Perfect agreement diagonal
    ax.plot([0, max_val], [0, max_val], ":", color="gray", linewidth=1)

    # Zone A boundaries
    ax.plot([0, 58.3],   [0, 70],           "-",  **kw_g)
    ax.plot([58.3, max_val], [70, max_val * 1.2], "-",  **kw_g)
    ax.plot([0, 70],     [0, 56],           "-",  **kw_g)
    ax.plot([70, max_val], [56, max_val * 0.8], "-",  **kw_g)

    # Zone C boundaries
    ax.plot([70, 290],   [180, 400],        **kw_r)
    ax.plot([130, 180],  [0, 70],           **kw_r)

    # Zone D / E boundaries
    ax.plot([0, 70],     [180, 180],        **kw_p)
    ax.plot([70, 70],    [180, max_val],    **kw_p)
    ax.plot([180, max_val], [70, 70],       **kw_p)
    ax.plot([180, 180],  [0, 70],           **kw_p)
    ax.plot([240, 240],  [70, 180],         **kw_p)
    ax.plot([240, max_val], [180, 180],     **kw_p)
    ax.plot([240, max_val], [70, 70],       **kw_p)

    # Zone labels
    label_kw = dict(fontsize=14, fontweight="bold")
    ax.text(30,  22,  "A", color="green",  **label_kw)
    ax.text(110, 60,  "B", color="orange", **label_kw)
    ax.text(55,  160, "D", color="purple", **label_kw)
    ax.text(45,  260, "E", color="purple", **label_kw)
    ax.text(320, 55,  "E", color="purple", **label_kw)
    ax.text(300, 130, "D", color="purple", **label_kw)
    ax.text(200, 330, "C", color="red",    **label_kw)
    ax.text(155, 28,  "C", color="red",    **label_kw)
    ax.text(310, 250, "B", color="orange", **label_kw)


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------
def clarke_error_grid(
    ref_values:  np.ndarray,
    pred_values: np.ndarray,
    title: str = "Clarke Error Grid Analysis",
    save_path: str | None = None,
) -> dict:
    """Plot a Clarke Error Grid and return zone summary dict.

    Args:
        ref_values  : Ground-truth blood glucose values (mg/dL).
        pred_values : Predicted blood glucose values (mg/dL).
        title       : Plot title.
        save_path   : PNG output path (default: clarke_error_grid.png next to script).

    Returns:
        Zone summary dict (same structure as summarize_clarke_zones).
    """
    assert len(ref_values) == len(pred_values), \
        f"Array length mismatch: refs={len(ref_values)}, preds={len(pred_values)}"

    if save_path is None:
        save_path = str(Path(__file__).parent / "clarke_error_grid.png")

    refs  = np.asarray(ref_values,  dtype=np.float32)
    preds = np.asarray(pred_values, dtype=np.float32)

    results = summarize_clarke_zones(refs, preds)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    max_val = max(400.0, float(refs.max()), float(preds.max())) + 10.0

    ax.scatter(refs, preds, marker="o", color="#1f77b4", s=22, alpha=0.65,
               label="Predictions", zorder=3)
    _draw_clarke_boundaries(ax, max_val)

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Reference Blood Glucose (mg/dL)", fontsize=13)
    ax.set_ylabel("Predicted Blood Glucose (mg/dL)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotation box
    ann_lines = [
        f"n = {results['total_points']}",
        f"Zone A: {results['zone_A_pct']:.1f}%",
        f"Zone B: {results['zone_B_pct']:.1f}%",
    ]
    ax.text(
        0.02, 0.97, "\n".join(ann_lines),
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("Clarke Error Grid — %d points", results["total_points"])
    for z in ZONE_ORDER:
        logger.info(
            "  Zone %s: %4d (%5.1f%%)",
            z, results[f"zone_{z}_count"], results[f"zone_{z}_pct"],
        )
    logger.info("Saved: %s", save_path)
    return results


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    rng  = np.random.default_rng(42)
    refs = rng.uniform(80, 200, 200).astype(np.float32)
    pred = refs + rng.normal(0, 8, 200).astype(np.float32)

    result = clarke_error_grid(refs, pred, title="Smoke-test Clarke Grid")
    print(json.dumps(result, indent=2))
    print("plot_clarke_error_grid.py  OK")
