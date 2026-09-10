# Training Outputs Walkthrough

## Overview

This file captures the full rerun walkthrough for the updated training pipeline:

- learning-rate scheduler added to `train_fusion_model.py`
- epoch metrics logged to `training_log.csv`
- Clarke grid summary saved to `clarke_results.json`
- `generate_extra_figures.py` updated to plot real loss curves from the CSV log

## Step 0: Environment Note

The workspace `.venv` did not contain `torch`, so the first attempted run failed with:

```text
ModuleNotFoundError: No module named 'torch'
```

The successful rerun used the existing Conda environment:

```text
/opt/homebrew/Caskroom/miniconda/base/envs/ai_core/bin/python
```

## Step 1: Fresh Retraining Run

Command used:

```bash
rm -f fusion_model_v2_mac.pth fusion_model_mac.pth training_log.csv clarke_results.json real_loss_curves.png clarke_error_grid.png && "/opt/homebrew/Caskroom/miniconda/base/envs/ai_core/bin/python" train_fusion_model.py
```

Console output:

```text
Hardware Acceleration Found: Apple Silicon GPU (MPS)

Loading Dataset and Splitting by Patient ID (No Leakage)...
Dataset initialized with 1409 continuous windows of size 500.
Dataset initialized with 280 continuous windows of size 500.

Starting Training Strategy Fusion Model (1D+2D+3D)...
Max Epochs: 100 | Batch Size: 32 | Training Samples: 1409

Epoch [1/100] | Time: 7.66s | Train MSE: 5641.4262 | Val MSE: 532.2495
  -> Validation loss improved! Saving model.

Epoch [2/100] | Time: 3.84s | Train MSE: 365.7183 | Val MSE: 138.9615
  -> Validation loss improved! Saving model.

Epoch [3/100] | Time: 3.73s | Train MSE: 199.9552 | Val MSE: 111.0016
  -> Validation loss improved! Saving model.

Epoch [4/100] | Time: 3.84s | Train MSE: 196.7881 | Val MSE: 83.1205
  -> Validation loss improved! Saving model.

Epoch [5/100] | Time: 3.87s | Train MSE: 194.4395 | Val MSE: 92.5322
  -> No improvement. Patience: 1/5

Epoch [6/100] | Time: 3.95s | Train MSE: 185.0931 | Val MSE: 133.1082
  -> No improvement. Patience: 2/5

Epoch [7/100] | Time: 3.97s | Train MSE: 183.3567 | Val MSE: 183.4201
  -> No improvement. Patience: 3/5

Epoch [8/100] | Time: 3.70s | Train MSE: 207.2031 | Val MSE: 179.1040
  -> No improvement. Patience: 4/5

Epoch [9/100] | Time: 3.81s | Train MSE: 183.0570 | Val MSE: 112.4523
  -> No improvement. Patience: 5/5

🛑 EARLY STOPPING TRIGGERED! Validation loss hasn't improved for 5 epochs.

✅ Training Complete! The OPTIMAL number of epochs was 4.
The best model has been saved to: /Users/namangarg/Desktop/Gaurav Verma/fusion_model_v2_mac.pth

Running Clinical Validation on Best Model (Clarke Error Grid)...
Total Points: 280
Zone A (Accurate): 280 (100.00%)
Zone B (Acceptable): 0 (0.00%)
Zone C+ (Dangerous): 0 (0.00%)
Saved plot to 'clarke_error_grid.png'
```

## Step 2: Full `training_log.csv`

```csv
epoch,train_mse,val_mse,lr
1,5641.426188439828,532.249462890625,0.001
2,365.7183341465578,138.96154174804687,0.00095
3,199.95521552990817,111.00162615094867,0.0009025
4,196.78812974289514,83.1204851422991,0.000857375
5,194.43950801982027,92.53222394670759,0.0008145062499999999
6,185.09305238148437,133.10818219866073,0.0007737809374999998
7,183.35667602452426,183.42007359095982,0.0007350918906249997
8,207.20310445572994,179.1040326799665,0.0006983372960937497
9,183.05696533181467,112.45227704729352,0.0006634204312890621
```

## Step 3: Full `clarke_results.json`

```json
{
  "zone_A_count": 280,
  "zone_B_count": 0,
  "zone_C_count": 0,
  "zone_D_count": 0,
  "zone_E_count": 0,
  "total_points": 280,
  "zone_A_pct": 100.0,
  "zone_B_pct": 0.0
}
```

## Step 4: Real Loss-Curve Figure Generation

Command used:

```bash
"/opt/homebrew/Caskroom/miniconda/base/envs/ai_core/bin/python" generate_extra_figures.py
```

Relevant code path now used by `generate_extra_figures.py`:

```python
log = pd.read_csv("training_log.csv")
epochs = log["epoch"].values
train_loss = log["train_mse"].values
val_loss = log["val_mse"].values
best_epoch = int(log.loc[log["val_mse"].idxmin(), "epoch"])
```

This confirms the figure was generated from `training_log.csv`, not from synthetic mathematical formulas.

Runtime note:

```text
/Users/namangarg/Desktop/Gaurav Verma/generate_extra_figures.py:41: SyntaxWarning: invalid escape sequence '\l'
    plt.xlabel("Signal-to-Noise Ratio (dB) $\leftarrow$ Higher Noise")
```

That warning comes from the SNR figure label and does not affect loading the real training log or saving `real_loss_curves.png`.

## Step 5: Generated Artifacts

Files produced by the rerun:

- `fusion_model_v2_mac.pth`
- `training_log.csv`
- `clarke_results.json`
- `clarke_error_grid.png`
- `real_loss_curves.png`

## Final Metrics Report

- Final epoch count: 9
- Best validation epoch: 4
- Final train MSE: 183.05696533181467
- Final val MSE: 112.45227704729352
- Zone A %: 100.0
- Zone B %: 0.0
- Total Clarke points: 280
- Zone A count: 280
- Zone B count: 0
- Zone C count: 0
- Zone D count: 0
- Zone E count: 0

## Honest Check: Is Validation MSE Ever Lower Than Training MSE?

Yes.

Validation MSE is lower than training MSE in 8 out of 9 epochs:

```text
Epochs: [1, 2, 3, 4, 5, 6, 8, 9]
```

Why this can happen here:

- the validation set is only 4 patients out of 22 total, so it is a small patient-level sample and can be easier than the broader training distribution
- training loss is measured while optimization noise and dropout are active, which can make training MSE look worse than evaluation-time validation MSE
- patient-to-patient variation can dominate the split; with only 4 held-out patients, the validation cohort may simply be less difficult than the 18-patient training cohort

This means `val_mse < train_mse` is not automatically suspicious in this setup, especially with a small holdout split.