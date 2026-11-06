# ppgbench results

Source: `results/_smoke.json`

## synthetic / subject_loso

| Model | n | MAE | RMSE | MARD % | R² | r | within-r | Zone A % | A+B % | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| Train mean (`mean`) | 144 | 20.8 | 26.0 | 13.2 | -2.40 | -0.86 | -- | 69.4 | 100.0 | 1 |
| Ridge (PPG features) (`ridge`) | 144 | 20.8 | 27.5 | 13.1 | -2.80 | -0.81 | -0.57 | 71.5 | 100.0 | 1 |
| 1-D CNN (`1d`) | 144 | 18.9 | 24.2 | 11.9 | -1.95 | -0.89 | -0.39 | 72.2 | 100.0 | 1 |

## Datasets

| Dataset | records | subjects | windows | glucose min | max | mean | std |
|---|---|---|---|---|---|---|---|
| synthetic | 288 | 6 | 1152 | 73 | 179 | 131.3 | 25.2 |

