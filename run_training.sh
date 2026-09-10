#!/usr/bin/env bash
# =============================================================================
# run_training.sh
# ===============
# One-command launcher for the PPG-Glucose Fusion Model training.
#
# Features
# --------
#   - Auto-detects the correct Python (conda ai_core env with torch)
#   - PYTHONUNBUFFERED=1 ensures live log output (no buffering)
#   - Tees stdout+stderr to a timestamped log file
#   - Prints Python path, torch version, and MPS/CUDA availability at startup
#   - Passes all CLI args through to train_fusion_model.py
#   - Reports final exit code and log path on exit
#
# Usage
# -----
#   chmod +x run_training.sh
#   ./run_training.sh                           # default 100 epochs
#   ./run_training.sh --epochs 50 --batch 16
#   ./run_training.sh --epochs 200 --lr 0.0005 --patience 15
#
# CLI args are forwarded to train_fusion_model.py:
#   --epochs    N       number of training epochs          (default: 100)
#   --batch     N       batch size                         (default: 32)
#   --lr        F       initial learning rate              (default: 0.001)
#   --patience  N       early-stopping patience            (default: 10)
#   --min-delta F       min improvement to reset patience  (default: 0.5)
#   --save      PATH    checkpoint save path               (default: fusion_model_v2_mac.pth)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script directory (works regardless of CWD)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Auto-detect Python with PyTorch
# ---------------------------------------------------------------------------
PYTHON=""

# 1. Try the .venv in the workspace
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
    if "$VENV_PY" -c "import torch" 2>/dev/null; then
        PYTHON="$VENV_PY"
        echo "[run_training.sh] Using .venv Python: $PYTHON"
    fi
fi

# 2. Try conda ai_core environment
if [[ -z "$PYTHON" ]]; then
    CONDA_PY="/opt/homebrew/Caskroom/miniconda/base/envs/ai_core/bin/python"
    if [[ -x "$CONDA_PY" ]]; then
        if "$CONDA_PY" -c "import torch" 2>/dev/null; then
            PYTHON="$CONDA_PY"
            echo "[run_training.sh] Using conda ai_core Python: $PYTHON"
        fi
    fi
fi

# 3. Try conda base environment
if [[ -z "$PYTHON" ]]; then
    CONDA_BASE_PY="/opt/homebrew/Caskroom/miniconda/base/bin/python"
    if [[ -x "$CONDA_BASE_PY" ]]; then
        if "$CONDA_BASE_PY" -c "import torch" 2>/dev/null; then
            PYTHON="$CONDA_BASE_PY"
            echo "[run_training.sh] Using conda base Python: $PYTHON"
        fi
    fi
fi

# 4. System python3 as last resort
if [[ -z "$PYTHON" ]]; then
    if command -v python3 &>/dev/null; then
        if python3 -c "import torch" 2>/dev/null; then
            PYTHON="$(command -v python3)"
            echo "[run_training.sh] Using system python3: $PYTHON"
        fi
    fi
fi

if [[ -z "$PYTHON" ]]; then
    echo ""
    echo "ERROR: Could not find a Python installation with PyTorch."
    echo ""
    echo "Install PyTorch in your environment and try again:"
    echo "  conda activate ai_core && pip install torch"
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Print environment info
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  PPG-Glucose Fusion Model — Training Launcher"
echo "============================================================"
echo "  Timestamp  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python     : $PYTHON"
echo "  Version    : $("$PYTHON" --version 2>&1)"
echo "  Torch      : $("$PYTHON" -c 'import torch; print(torch.__version__)')"
echo "  MPS avail  : $("$PYTHON" -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'n/a')"
echo "  CUDA avail : $("$PYTHON" -c 'import torch; print(torch.cuda.is_available())')"
echo "  Script dir : $SCRIPT_DIR"
echo "  CLI args   : ${*:-<none>}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Set up log file
# ---------------------------------------------------------------------------
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$SCRIPT_DIR/training_run_${TIMESTAMP}.log"

echo "Live log: $LOG_FILE"
echo ""

# ---------------------------------------------------------------------------
# Run training — live output to console AND log file
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

{
    "$PYTHON" -u train_fusion_model.py "$@"
} 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT="${PIPESTATUS[0]}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
if [[ "$TRAIN_EXIT" -eq 0 ]]; then
    echo "  Training completed successfully."
else
    echo "  Training FAILED (exit code: $TRAIN_EXIT)"
fi
echo "  Log saved to: $LOG_FILE"
echo "  Finished at : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

exit "$TRAIN_EXIT"
