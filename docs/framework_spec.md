# ppgbench — specification (v0.1)

A leakage-controlled benchmark framework for regressing a physiological target
(blood glucose) from wearable optical signals (PPG / BVP).  Every result it
produces is (i) evaluated under an explicit, named split protocol with a
machine-checked no-leak guarantee, (ii) reported at the level of independent
reference measurements, (iii) accompanied by null and classical baselines and
cluster-bootstrap confidence intervals, and (iv) exportable to LaTeX/JSON so a
paper never carries a hand-typed number.

The package lives in `ppgbench/` next to the existing scripts.  The existing
scripts (`ppg_data.py`, `run_subject_cv.py`, `run_bigideas_cv.py`,
`run_leakage_ladder.py`, `bootstrap_ci.py`, `make_results_macros.py`,
`make_paper_figures.py`, `baseline_models.py`, `fusion_network_model.py`,
`plot_clarke_error_grid.py`) are the reference behaviour: the package must
reproduce their numbers (same preprocessing, same splits, same metrics).  Read
them before implementing.  Do not modify them; the maintainer will turn them
into thin wrappers once the package passes its tests.

Python 3.12, numpy, scipy, pandas, scikit-learn, torch (MPS/CPU), pywt,
opencv-headless, neurokit2 (optional, guarded import), matplotlib.  No other
dependencies.  Interpreter: `~/.venvs/ppgfusion/bin/python`.  **Do not run
training longer than ~10 s during development** (GPU jobs are running); use
the `--quick` paths and tiny synthetic data in tests.

## Package layout and ownership

```
ppgbench/
  __init__.py            version, lazy re-exports                    (owner: core)
  core/
    windowset.py         WindowSet dataclass + helpers              (core)
    signal.py            bandpass, resample, windows, zscore, sqi   (core)
    views.py             cwt_scalogram, phase_space, make_views     (core)
    features.py          handcrafted + optional neurokit features   (core)
  data/
    registry.py          @register_dataset / load_dataset(name, **) (data)
    mendeley.py          raw Mendeley loader                        (data)
    bigideas.py          BIG IDEAs npz loader + fetch wrapper       (data)
    synthetic.py         deterministic synthetic dataset for tests  (data)
  protocols.py           split protocols + no-leak assertions       (protocols)
  models/
    registry.py          @register_model / build_model(name)        (models)
    classical.py         mean, subject_mean, ridge, gbm (+demo)     (models)
    nets.py              SingleBranch, Fusion wrapper, ResNet1D, CNNLSTM, TinyTransformer (models)
    pretrained.py        Encoder protocol + ProbeModel (frozen encoder + ridge/MLP); PaPaGei adapter stub (models)
  train.py               train_fold(...) with fixed budget, target standardisation (models)
  metrics.py             mae/rmse/mard/r2/pearson/clarke/parkes/within_r, record_level, cluster_bootstrap (metrics)
  integrity.py           detect_label_cycling, detect_duplicate_windows, assert_no_leak, check_timestamps (metrics)
  report.py              ResultStore (JSON), to_latex_macros, to_markdown, figure helpers (report)
  cli.py                 `python -m ppgbench ...` (report)
  __main__.py            -> cli.main()                               (report)
tests/ppgbench/
  test_core.py test_data.py test_protocols.py test_models.py test_metrics.py test_integrity.py test_report.py test_cli.py
docs/framework.md        user guide (report)
```

Ownership tags say which implementation agent writes which files.  Agents must
not edit files they do not own; shared interfaces are defined below and are
binding.

## Shared interfaces (binding)

### WindowSet (`ppgbench/core/windowset.py`)
```python
@dataclass
class WindowSet:
    X: np.ndarray            # (N, L) float32, z-scored per window
    y: np.ndarray            # (N,) float32 target (mg/dL)
    subject: np.ndarray      # (N,) str   split key for subject-wise protocols
    record: np.ndarray       # (N,) str   one independent reference measurement
    time: np.ndarray | None  # (N,) float64 POSIX seconds of the reference (None if unknown)
    meta: pd.DataFrame       # one row per record: record, subject, glucose, [age, sex, height, weight, egv_time ...]
    fs: int = 64
    source: str = ""
    def __len__(self); def subset(self, mask_or_idx) -> "WindowSet"; def describe(self) -> str
    def demographics(self, cols=("age","height","weight"), sex_col="sex") -> np.ndarray | None  # (N,k) or None
```

### Signal (`ppgbench/core/signal.py`)
`bandpass(x, fs, lo=0.5, hi=8.0, order=4)`, `to_rate(x, fs_in, fs_out=64)` (band-pass at native rate then `resample_poly`), `sliding_windows(x, win, hop)`, `zscore_rows(X)`, `sqi(X, fs)` → per-window quality dict/array (std, flatline fraction, dominant-frequency plausibility 0.7–3 Hz, skewness).  Must match `ppg_data.py` numerically.

### Views (`ppgbench/core/views.py`)
`cwt_scalogram(x, n_scales=64, size=64)`, `phase_space(x, tau=5, n_points=256)`, `make_views(X) -> dict[str, torch.Tensor]` with keys `x1 (N,1,L)`, `x2 (N,1,64,64)`, `x3 (N,3,256)`.  Must match `ppg_data.make_views` numerically (same functions).

### Features (`ppgbench/core/features.py`)
`handcrafted_features(X, fs=64) -> (N, 9)` identical to `ppg_data.handcrafted_features`; `morphology_features(X, fs=64)` using neurokit2 if importable (peaks, rise time, pulse width, dicrotic notch stats, HRV from peak intervals), else raises `ImportError` with a clear message; `FEATURE_NAMES` lists.

### Datasets (`ppgbench/data/registry.py`)
```python
def register_dataset(name): ...          # decorator
def load_dataset(name: str, **kwargs) -> WindowSet
def list_datasets() -> list[str]
```
`mendeley` (kwargs: root, win, hop), `bigideas` (dir_path, subjects, lookback_s, max_windows_per_egv), `synthetic` (n_subjects, records_per_subject, windows_per_record, seed; target = subject offset + small signal-dependent term so models CAN learn something; `time` monotone per subject).  Loaders must reproduce `ppg_data.load_mendeley` / `load_bigideas` (`time` = `egv_time` for bigideas, None for mendeley).

### Protocols (`ppgbench/protocols.py`)
```python
Fold = tuple[np.ndarray, np.ndarray]           # train idx, test idx
def make_folds(ws: WindowSet, protocol: str, seed: int = 0, **kw) -> list[Fold]
PROTOCOLS = {
  "window_random":  KFold(5, shuffle) over windows                       # leakiest
  "record_random":  5 folds grouped by record (subjects shared)
  "subject_kfold":  GroupKFold(k) by subject
  "subject_loso":   LeaveOneGroupOut by subject
  "chronological":  per subject, first train_frac (0.7) of records by time -> train, rest test, guard gap gap_s (3600)
  "chronological_nogap": same with gap_s=0
  "personal":       chronological, one fold per subject restricted to that subject (skip subjects with < min_train windows)
}
def leak_report(ws, folds, protocol) -> dict   # counts of shared windows/records/subjects/time-overlap between train and test per fold
def assert_no_leak(ws, folds, level: str)      # level in {"window","record","subject","time"}; raises LeakError with a precise message
```
Every protocol's folds must pass `assert_no_leak` at its declared level (`window_random` → "window"; `record_random` → "record"; subject_* → "subject"; chronological* / personal → "subject"? no: they share subjects by design, so level "time" meaning no test record's time lies within [train_min, train_max + gap] of the same subject).  Reproduce `run_subject_cv.py` LOSO and `run_leakage_ladder.py` / `run_bigideas_cv.py` splits exactly.

### Models (`ppgbench/models/registry.py`)
```python
@register_model(name, kind)   # kind in {"classical", "net", "probe"}
def build_model(name, **kw)
def list_models() -> dict[str, str]
```
Classical models implement `fit(ws, tr_idx, feats: dict) -> self; predict(ws, te_idx, feats) -> np.ndarray` where `feats` = {"hand": (N,9), "demo": (N,k) or None, "morph": optional}.  Nets are `torch.nn.Module` with `forward(x1, x2, x3)` (unused inputs ignored).  Probe models wrap an `Encoder` (`embed(X: np.ndarray, fs) -> np.ndarray (N,d)`) with ridge or small MLP on top; `pretrained.py` defines the `Encoder` protocol, an `IdentityEncoder` (handcrafted features) for tests, and a `PaPaGeiEncoder` adapter that raises a clear `NotImplementedError`/`ImportError` if weights are not present (path via env `PPGBENCH_PAPAGEI`).  `TinyTransformer`: patch-embed 1-D (patch 16) + 2 encoder layers (d=64, 4 heads) + mean-pool + linear.

### Training (`ppgbench/train.py`)
`train_net(name, views, y, tr, te, epochs, batch, lr, seed, device, **model_kw) -> np.ndarray` reproducing `run_subject_cv.train_fold` exactly (AdamW wd 1e-4, OneCycle pct 0.2, smooth-L1 on standardised target, grad-clip 1, drop_last).  `device()` helper.

### Metrics (`ppgbench/metrics.py`)
`record_level(pred, ws) -> DataFrame[record, subject, y, pred, n_windows]`; `summarize(df) -> dict` (n, mae, rmse, mard_pct, r2, pearson_r, within_subject_r_median, zone_A..E_pct, parkes_zone_A..E_pct for type-1 consensus grid); `clarke_zone(ref, pred)`, `parkes_zone(ref, pred, diabetes_type=1)`; `cluster_bootstrap(df_model, df_null, B=5000, seed=0) -> dict` identical to `bootstrap_ci.py`.  Degenerate inputs (n<2 or zero variance) return a dict with `error="degenerate"` instead of raising.

### Integrity (`ppgbench/integrity.py`)
`detect_label_cycling(labels: np.ndarray, max_period=32) -> dict(period, fraction_matching)` (returns period k with fraction of rows where label[i]==label[i-k] > 0.98, else None); `detect_duplicate_windows(X, tol=1e-6) -> list[(i,j)]` (hash-based, exact duplicates and shifted overlaps for a given hop); `check_timestamps(t, expected_spacing_s=300, tol=5)`; `assert_no_leak` re-exported from protocols; `audit(ws, folds, protocol) -> dict` combining all.

### Report (`ppgbench/report.py`)
`ResultStore(path)` with `add(key, metrics_dict)`, `get`, `items`, atomic JSON write; `to_latex_macros(store, prefix_map) -> str` reproducing `make_results_macros.py` naming (`\mendFusionMAE` etc., `\ensuremath{\pm}` for multi-seed); `to_markdown(store) -> str`; `fig_clarke(ax, y, pred)`, `fig_model_bars(ax, ...)`, `fig_tracking(ax, df_subject)` using the palette in `make_paper_figures.py` (`#2a78d6, #eb6834, #1baf7a, #eda100`, ink `#0b0b0b`).

### CLI (`ppgbench/cli.py`)
```
python -m ppgbench datasets                      # list
python -m ppgbench models
python -m ppgbench run --dataset mendeley --protocol subject_loso --models mean ridge 1d fusion --seeds 0 1 2 --epochs 40 --out results/ppgbench.json [--quick]
python -m ppgbench ladder --dataset mendeley --models mean subject_mean 1d --protocols window_random record_random subject_loso
python -m ppgbench audit --dataset mendeley --protocol subject_loso          # leak report + integrity
python -m ppgbench report --store results/ppgbench.json --latex results_macros_ppgbench.tex --markdown results/summary.md
```
`run` prints one line per (model, seed) and writes `results/<store>` plus per-run record-level CSVs; `--quick` uses 3 folds and 2 epochs.

## Quality bar
- Type hints, docstrings with the scientific rationale, logging via `logging.getLogger("ppgbench.<module>")`.
- Deterministic: seeds threaded through; no `random` without seed.
- Tests use the `synthetic` dataset only (fast, < 20 s total); numeric parity tests against the legacy scripts may be marked `@pytest.mark.slow` and skipped unless `PPGBENCH_SLOW=1`.
- Every protocol is covered by a leak test that constructs a deliberately leaky split and asserts `LeakError` is raised, and a correct split and asserts it passes.
- No file outside the owner's list is modified.  No commits.
