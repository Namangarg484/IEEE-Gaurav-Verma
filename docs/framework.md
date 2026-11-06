# ppgbench — user guide

`ppgbench` is a leakage-controlled benchmark framework for regressing a
physiological target (blood glucose, mg/dL) from a single wearable optical
channel (PPG / BVP).  It grew out of the paper *"How Much Glucose Information
Does a Single PPG Channel Carry?"* and packages the evaluation discipline that
paper needed — named split protocols with machine-checked no-leak guarantees,
record-level metrics, null and classical baselines, cluster-bootstrap
confidence intervals and an export path that leaves no hand-typed number in a
manuscript — so that the next model, dataset or protocol is evaluated under
exactly the same rules as the last one.

The formal interface contract is `docs/framework_spec.md`; this document is the
user guide.

---

## 1. What the framework guarantees

Every number that comes out of `python -m ppgbench run` satisfies all of the
following.  None of them can be switched off from the command line.

| Guarantee | Where it is enforced |
|---|---|
| **A named protocol with a declared leak level.** Folds are built only by `ppgbench.protocols.make_folds(ws, protocol, seed)`. Each protocol declares the level it guarantees (`window`, `record`, `subject` or `time`) and `assert_no_leak` raises `LeakError` with a precise message if a fold violates it. `python -m ppgbench audit` runs that check plus the integrity audit and exits non-zero on a leak. | `protocols.py`, `integrity.py`, `cli.cmd_audit` |
| **Record-level reporting.** Window predictions are averaged per *record* (one independent reference measurement: a Mendeley recording, a CGM reading) by `metrics.record_level` before any metric is computed. There is no code path in the CLI that reports window-level numbers; the per-fold window MAE in the log is diagnostic only. | `metrics.py`, `cli.Experiment.run` |
| **Null and classical baselines under the identical pipeline.** `mean` (train mean), `subject_mean` (each subject's own train mean — the right null for chronological protocols) and ridge / GBM on nine hand-crafted PPG features share the fold indices, the feature cache and the metric code with the networks. | `models/classical.py`, `cli.Experiment` |
| **Fixed training budget, no peeking.** Networks train for a fixed epoch budget with AdamW + OneCycle on a target standardised with *train* statistics only; there is no early stopping on the held-out fold. `train_net` reproduces the legacy `run_subject_cv.train_fold` bit for bit. | `train.py` |
| **Determinism.** Seeds are threaded into fold construction (`window_random`), model initialisation and the data order; the synthetic dataset is a pure function of its arguments. Two runs with the same arguments produce the same store. | everywhere |
| **Uncertainty at the right level.** `cluster_bootstrap` resamples *subjects* (the independent unit), pairs model and null on the same resample, and reports the MAE CI, the paired-difference CI and P(model better than null). | `metrics.py`, `cli.cmd_bootstrap` |
| **No hand-typed numbers.** Results live in one JSON store with atomic writes, next to the record-level prediction CSVs they were computed from; `report` turns the store into `\newcommand` macros, Markdown and figures. The macro names reproduce the legacy `make_results_macros.py` exactly, so the paper compiles unchanged. | `report.py`, `cli.cmd_report` |
| **Integrity checks for the failure modes that invalidated the first draft.** `integrity.audit` detects cycling labels (the corrupted derivative CSV), duplicated / shifted windows and mis-spaced timestamps before anything is trained. | `integrity.py` |
| **Smoke-test runs are marked.** `--quick` (first 3 folds, 2 epochs) exists so the whole pipeline is exercised in seconds; such entries carry `"quick": true` and must never be reported. | `cli.py` |

---

## 2. Installation and environment

```bash
PY=~/.venvs/ppgfusion/bin/python     # Python 3.12; numpy, scipy, pandas, scikit-learn, torch (MPS/CPU), pywt,
                                     # opencv-headless, matplotlib; neurokit2 optional (morphology features)
cd "<repository root>"
$PY -m ppgbench --help
$PY -m pytest tests/ppgbench -q      # synthetic-data tests, < 1 min
PPGBENCH_SLOW=1 $PY -m pytest tests/ppgbench -q -m slow    # numeric parity against the legacy scripts (needs the datasets)
```

The package is imported from the repository root (no installation step).
Raw data locations follow the legacy scripts: Mendeley under
`Archive_Dataset/archive (2)/PPG_Dataset`, BIG IDEAs extracts under
`data.nosync/bigideas/sub_XXX.npz` (produced by `scripts/fetch_bigideas.py`).
Both can be overridden with `--dataset-arg root=...` / `--dataset-arg dir_path=...`.

---

## 3. Quick start

```bash
$PY -m ppgbench datasets                       # mendeley  bigideas  synthetic
$PY -m ppgbench models                         # name + kind (classical | net | probe)
$PY -m ppgbench protocols                      # name + guaranteed leak level

# smoke test of the whole pipeline on the synthetic corpus (seconds)
$PY -m ppgbench run --dataset synthetic --quick --models mean ridge 1d --out /tmp/pb/ppgbench.json

# the paper's main experiment: leave-one-subject-out on Mendeley, three seeds per network
$PY -m ppgbench run --dataset mendeley --protocol subject_loso \
    --models mean ridge demo ridge+demo 1d 2d 3d fusion --seeds 0 1 2 --epochs 40

# the leakage ladder (same models, progressively less leaky protocols)
$PY -m ppgbench ladder --dataset mendeley --models mean subject_mean 1d fusion --seeds 0 1 2
$PY -m ppgbench ladder --dataset bigideas --models mean subject_mean 1d --epochs 15 --batch 64

# leak report + integrity audit of a split (exit code 1 on a leak)
$PY -m ppgbench audit --dataset mendeley --protocol subject_loso

# cluster-bootstrap CIs of every model against the null
$PY -m ppgbench bootstrap --dataset mendeley --protocol subject_loso
$PY -m ppgbench bootstrap --dataset bigideas --protocol chronological     # null = subject_mean automatically

# export
$PY -m ppgbench report --latex results_macros_ppgbench.tex --markdown results/summary.md --figures figures/ppgbench
```

Every subcommand accepts `-v` (info), `-vv` (debug) or `-q` (warnings only),
before or after the subcommand name.

---

## 4. Concepts

### 4.1 `WindowSet`

The single data container (`ppgbench.core.windowset.WindowSet`):

| field | shape | meaning |
|---|---|---|
| `X` | `(N, L)` float32 | z-scored windows (5 s at 64 Hz → `L = 320`), band-passed 0.5–8 Hz at the native rate and resampled |
| `y` | `(N,)` float32 | reference glucose of the window's record (mg/dL) |
| `subject` | `(N,)` str | split key for subject-wise protocols |
| `record` | `(N,)` str | one independent reference measurement; the unit of reporting |
| `time` | `(N,)` float64 or `None` | POSIX seconds of the reference (`egv_time` for BIG IDEAs; `None` for Mendeley) |
| `meta` | DataFrame | one row per record (`record, subject, glucose, [age, sex, height, weight, egv_time, ...]`) |
| `fs`, `source` | | sampling rate, provenance string |

`ws.subset(mask_or_idx)`, `ws.describe()` and `ws.demographics()` (the `(N, k)`
matrix the `demo` models use, or `None`) are the only methods the CLI relies on.

### 4.2 Protocols and their leak levels

| protocol | folds | guaranteed level | what it answers |
|---|---|---|---|
| `window_random` | 5-fold `KFold(shuffle, random_state=seed)` over windows | `window` | the leakiest rung: windows of the same recording on both sides |
| `record_random` | 5 folds grouped by record | `record` | records separated, subjects shared |
| `record_kfold` | `GroupKFold(5)` by record (legacy BIG IDEAs "reading" rung) | `record` | neighbouring CGM readings on both sides |
| `subject_kfold` | `GroupKFold(n_splits)` by subject | `subject` | cross-person generalisation, k folds |
| `subject_loso` | `LeaveOneGroupOut` by subject | `subject` | the honest cross-subject protocol (paper) |
| `chronological_nogap` | per subject, first 70 % of records by time → train, rest → test | `time` | within-subject tracking without a guard gap |
| `chronological` | same with a guard gap (`gap_s=3600`) | `time` | within-subject tracking, honest (paper) |
| `chronological_blocked` | blocked chronological variant | `time` | sensitivity to the cut |
| `personal` | chronological, one fold per subject restricted to that subject | `time` | one model per person |

Level `time` means: no test record of a subject lies within
`[train_min, train_max + gap]` of that subject's training records.  The
chronological protocols share subjects *by design*; the null model that makes
them interpretable is `subject_mean`, and `report` / `bootstrap` pick it
automatically for them (`NULL_MODEL_BY_PROTOCOL` in `report.py`).

Protocol parameters are passed with `--protocol-arg KEY=VALUE`
(`n_splits=5`, `gap_s=1800`, `train_frac=0.7`, ...); the values are recorded
in each store entry under `fold_kwargs`.

### 4.3 Models

Three kinds, all registered under a name in `ppgbench.models.registry`:

* **classical** — `fit(ws, tr_idx, feats) -> self; predict(ws, te_idx, feats) -> (n_te,)`
  with `feats = {"hand": (N, 9), "demo": (N, k) | None, "morph": optional}`.
  `mean`, `subject_mean`, `ridge`, `demo`, `ridge+demo`, `gbm`, `gbm+demo`, `*+morph`.
* **net** — `torch.nn.Module` with `forward(x1, x2, x3)` taking the three
  views (`x1 (N,1,L)` raw signal, `x2 (N,1,64,64)` CWT scalogram,
  `x3 (N,3,256)` Takens point cloud); unused inputs are ignored.
  `1d`, `2d`, `3d`, `fusion`, `resnet1d`, `cnnlstm`, `tiny_transformer`.
* **probe** — a frozen `Encoder` (`embed(X, fs) -> (N, d)`) with a ridge or
  small MLP head: `probe_ridge`, `probe_mlp`, `probe_raw`, `papagei_probe`
  (needs weights at `$PPGBENCH_PAPAGEI`; raises a clear error otherwise).
  Probes expose the classical `fit/predict` interface, so the CLI treats them
  like classical models.

Networks are trained once per seed (`--seeds 0 1 2`); classical and probe
models are deterministic and run once (seed 0), exactly as the legacy scripts
did.  Model keyword arguments go through `--model-arg KEY=VALUE`.

### 4.4 Training budget

`train_net(name, views, y, tr, te, epochs, batch, lr, seed, device)`:
AdamW (weight decay 1e-4), OneCycle (pct_start 0.2), smooth-L1 on the
standardised target, gradient clipping at 1.0, `drop_last` batches
(BatchNorm needs more than one sample).  Dataset defaults mirror the legacy
scripts: Mendeley 40 epochs / batch 16, BIG IDEAs 15 epochs / batch 64;
`--epochs`, `--batch`, `--lr` override them, `--quick` sets 2 epochs unless
`--epochs` is given.

### 4.5 Metrics

`metrics.summarize(df)` on the record-level table returns `n, mae, rmse,
bias, mard_pct, r2, pearson_r, within_subject_r_median`, Clarke zones
`zone_A..E_pct`, Parkes (type-1 consensus) zones `parkes_zone_A..E_pct` and
hypo/hyper event metrics.  Degenerate inputs (fewer than two records or a
constant reference) return `{"n", "mae", "error": "degenerate"}` instead of
raising, so a tiny `--quick` run still completes.  Per-subject summaries
(subjects with more than five records) are stored under `per_subject`.

---

## 5. The result store

One JSON file (default `results/ppgbench.json`), written atomically after
every run.  Keys:

| key | content |
|---|---|
| `<dataset>/<protocol>[_<variant>]/<model>/seed<k>` | the metrics dict of one run plus provenance: `dataset, protocol, base_protocol, variant, model, kind, seed, folds, quick, subjects, epochs/batch/lr/device (nets), fold_kwargs, protocol_params, gap_s, level, leak_checked, fold_fingerprint, seconds`.  A run with weakened protocol parameters (e.g. `--protocol-arg gap_s=0` or `train_frac=0.5`) is stored under `chronological_gap0` / `chronological_cut50`, never under the canonical key, and its null model and LaTeX macro name follow the base protocol. |
| `dataset/<name>` | corpus statistics at record level (`n_records, n_subjects, n_windows, glucose_min/max/mean/std, fs, window_samples`) |
| `bootstrap/<dataset>/<protocol>/<model>` | `cluster_bootstrap` output (`mae, mae_ci, diff_vs_null, diff_ci, p_better_than_null, n_subjects, n_records, null_model, B, seed`) |

Record-level predictions of each run are written next to the store:
`results/ppgbench_preds/<dataset>__<protocol>__<model>__seed<k>.csv` with
columns `record, subject, y, pred, n_windows[, time]` — the legacy layout, so
`bootstrap_ci.py`-style analyses still read them.

Python API:

```python
from ppgbench.report import ResultStore, make_key, aggregate, to_latex_macros, to_markdown
store = ResultStore("results/ppgbench.json")
store.experiments()                                  # [("mendeley", "subject_loso"), ...]
store.groups("mendeley", "subject_loso")             # {model: [metrics per seed]}
aggregate(store, "mendeley", "subject_loso")         # {model: (mean MAE, std, n_seeds)}
store.load_predictions("mendeley", "subject_loso", "fusion")   # seed-averaged record-level table
open("results_macros.tex", "w").write(to_latex_macros(store))
```

---

## 6. Exporting

### 6.1 LaTeX macros

`to_latex_macros(store, prefix_map=None)` emits, for every experiment
(`dataset/protocol` pair) with prefix `P` and every model token `M`:

| macro | value |
|---|---|
| `\P M MAE`, `RMSE`, `MARD`, `Rtwo`, `R`, `ZoneA`, `ZoneAB`, `WithinR` | metric; several seeds → `mean\ensuremath{\pm}std` |
| `\P M ParkesA`, `ParkesAB`, `Bias` | new: emitted when the metrics module reports them |
| `\P M Seeds`, `\P M N` | number of seeds, number of records |
| `\P M Wins` / `Worse` | subjects on which the model beats the null's per-subject MAE |
| `\P NSubjectsAll`, `\P Epochs` | |
| `\D NRecords`, `NSubjects`, `NWindows`, `GlucoseMin/Max/Mean/Std` | corpus statistics, `D` = dataset prefix |
| `\P M Diff`, `DiffCI`, `MAECI`, `PBetter` | bootstrap entries |
| `\D Ladder <Protocol> M` | MAE of every (dataset, protocol, model) group |

Default prefixes: `mendeley/subject_loso → mend`, `bigideas/chronological → big`,
`bigideas/subject_loso → bigLoso`; any other pair gets
`<datasetPrefix><ProtocolToken>` (e.g. `syntheticSubjectLoso`).  Model
tokens: `mean → Mean, ridge → Ridge, demo → Demo, ridge+demo → RidgeDemo,
1d → OneD, 2d → TwoD, 3d → ThreeD, fusion → Fusion, subject_mean → SubjMean,
gbm → Gbm, resnet1d → ResNet, cnnlstm → CnnLstm, tiny_transformer →
TinyTransformer`; unknown names are CamelCased with digits spelled out so
the macro is always legal LaTeX.  Protocol tokens: `window_random → Window,
record_random → Record, record_kfold → Reading, subject_loso → Subject,
chronological_nogap → ChronoNoGap, chronological → Chrono, personal →
Personal`.  Override with `--prefix mendeley/subject_loso=mend`.

`tests/ppgbench/test_report.py` pushes the same numbers through the legacy
`make_results_macros.py` and asserts the macro sets are identical.

### 6.2 Markdown and figures

`report --markdown` writes one table per experiment (plus bootstrap and
corpus tables).  `report --figures DIR` writes, per experiment,
`<dataset>_<protocol>_models.{pdf,png}` (MAE bars), `..._clarke_<model>`
(Clarke grid of the seed-averaged predictions) and, for chronological
protocols, `..._tracking_<model>` (reference vs. prediction over one
subject's held-out period).  The helpers `fig_clarke`, `fig_model_bars`,
`fig_tracking`, `fig_ladder` draw on a caller-supplied axis with the paper
palette (`#2a78d6 #eb6834 #1baf7a #eda100`, ink `#0b0b0b`), so composite
paper figures are built from the same primitives.

---

## 7. How to add a dataset

Built-in datasets: `mendeley` (Kermani–Esmaeili finger PPG, 23 subjects),
`bigideas` (PhysioNet wrist BVP vs CGM, 16 subjects, timestamps), `stuba`
(STUBA ePPG six-channel finger PPG, 24 subjects incl. 3 T1DM; `channel=`
selects `R1/IR1/G1/R2/IR2/G2` or `mean`) and `synthetic`.  Frozen folds for
every built-in dataset and protocol are exported by
`python scripts/freeze_folds.py` to `results/folds/<dataset>__<protocol>__seed<k>.json`
(record ids per fold plus the protocol's leak report and fingerprint);
a run whose `fold_fingerprint` differs from the frozen one was not evaluated
on the benchmark's folds.

1. Write a loader in `ppgbench/data/<name>.py` that returns a `WindowSet`:

   ```python
   from ppgbench.core.signal import to_rate, sliding_windows, zscore_rows, FS, WIN, HOP
   from ppgbench.core.windowset import WindowSet
   from ppgbench.data.registry import register_dataset

   @register_dataset("mydataset", description="...", citation="...", license="...", requires_files=True)
   def load_mydataset(root: str = DEFAULT_ROOT, win: int = WIN, hop: int = HOP) -> WindowSet:
       ...  # read raw signals, to_rate() at the native fs, sliding_windows(), zscore_rows()
       return WindowSet(X, y, subject, record, time, meta, fs=FS, source="mydataset")
   ```

   Rules: one `record` id per independent reference measurement (never per
   window); `subject` is the person; fill `time` with the reference's POSIX
   time whenever the dataset has one (chronological protocols need it), else
   `None`; `meta` has one row per record with at least `record, subject,
   glucose`, plus `age, sex, height, weight` if demographic baselines should
   run.  Use only the pre-reference part of a signal (no future).  Keep
   loader keyword arguments literal-parsable so `--dataset-arg` works.
2. Import the module from `ppgbench/data/__init__.py` so registration happens
   on `load_dataset` (see how `mendeley.py` and `bigideas.py` are listed).
3. Validate: `load_dataset("mydataset")` runs `validate_windowset`
   (shapes, dtypes, one glucose per record, monotone time per subject).
   Then `python -m ppgbench audit --dataset mydataset --protocol subject_loso`
   and `--protocol chronological` if `time` is set.
4. Add tests in `tests/ppgbench/test_data.py` on a tiny fixture directory
   (counts, record/label mapping, `time` alignment), plus a `@pytest.mark.slow`
   parity test if there is a legacy loader.
5. Optional: add macro prefixes in `report.DATASET_PREFIXES` /
   `DEFAULT_PREFIX_MAP` (otherwise the auto-generated prefix is used) and a
   ladder default in `cli.LADDER_PROTOCOLS`.

---

## 8. How to add a model

**Classical / probe** (`ppgbench/models/classical.py` or `pretrained.py`):

```python
from ppgbench.models.registry import register_model

@register_model("my_ridge", "classical")
class MyRidge:
    def fit(self, ws, tr, feats):          # feats["hand"] (N,9), feats["demo"] (N,k) | None
        ...; return self
    def predict(self, ws, te, feats):      # -> (len(te),) float32 mg/dL
        ...
```

Fit only on `tr`; never touch `ws.y[te]`.  Everything that must be learned
(scalers, PCA, feature selection) is fitted inside `fit`.

**Network** (`ppgbench/models/nets.py`):

```python
@register_model("my_net", "net")
class MyNet(nn.Module):
    def forward(self, x1, x2, x3):         # use any subset of the views; return (B, 1)
        ...
```

Nets are instantiated by `train_net` per fold and seed; constructor keywords
arrive through `--model-arg`.  Keep BatchNorm in mind: batches are
`drop_last`, and the CLI shrinks the batch when a tiny fold is smaller than
it.

**Probe with a new encoder** (`pretrained.py`): implement `embed(X, fs) ->
(N, d)` (the `Encoder` protocol), raise `PretrainedWeightsMissing` with the
download instructions when weights are absent, and register a `ProbeModel`
subclass or alias.

Then: add the display name to `report.MODEL_LABELS` and the macro token to
`report.MODEL_NAMES` if the automatic CamelCase is not what the paper should
say; add a test in `tests/ppgbench/test_models.py` (shape, determinism,
`fit`-does-not-touch-test); run `python -m ppgbench run --dataset synthetic
--quick --models my_net`.

---

## 9. How to add a protocol

```python
from ppgbench.protocols import register_protocol, Fold

@register_protocol("subject_blocked", level="subject", doc="GroupKFold by subject, contiguous blocks")
def subject_blocked(ws, seed=0, *, n_splits: int = 5) -> list[Fold]:
    ...  # return [(train_idx, test_idx), ...] as int arrays over windows
```

The declared `level` is a promise: `make_folds` (and `audit`) verify every
fold with `assert_no_leak(ws, folds, level)`, and `declared_level()` is what
the CLI records in the store.  Keyword-only parameters become `--protocol-arg`
options.  Requirements:

* a leak test in `tests/ppgbench/test_protocols.py` that builds a
  deliberately leaky split and asserts `LeakError`, and a correct split that
  passes;
* a macro token in `report.PROTOCOL_MACRO_NAMES` (else CamelCase is used)
  and, if the protocol shares subjects by design, its null model in
  `report.NULL_MODEL_BY_PROTOCOL`;
* if it is a rung of a leakage ladder, add it to `cli.LADDER_PROTOCOLS`.

---

## 10. Reproducing the paper

The legacy scripts remain the reference; the framework reproduces them
(parity tests under `PPGBENCH_SLOW=1`).  Mapping:

| legacy | ppgbench |
|---|---|
| `run_subject_cv.py --models ... --seeds 0 1 2` | `run --dataset mendeley --protocol subject_loso --models ... --seeds 0 1 2 --epochs 40` |
| `run_bigideas_cv.py --protocol within` | `run --dataset bigideas --protocol chronological --models subject_mean ridge 1d 2d 3d fusion --epochs 15 --batch 64 --dataset-arg max_windows_per_egv=2` |
| `run_bigideas_cv.py --protocol loso` | `run --dataset bigideas --protocol subject_loso ...` |
| `run_leakage_ladder.py --dataset mendeley` | `ladder --dataset mendeley --models mean subject_mean 1d fusion --seeds 0 1 2` |
| `run_leakage_ladder.py --dataset bigideas` | `ladder --dataset bigideas --models mean subject_mean 1d --epochs 15 --batch 64` |
| `bootstrap_ci.py` | `bootstrap --dataset mendeley --protocol subject_loso` and `bootstrap --dataset bigideas --protocol chronological` |
| `make_results_macros.py` | `report --latex results_macros_ppgbench.tex` |
| `make_paper_figures.py` | `report --figures figures/ppgbench` (+ the `fig_*` helpers for composite panels) |

Full sequence (M4 MacBook Air: ~20 min per Mendeley seed per network,
~15 min for the BIG IDEAs networks at 15 epochs):

```bash
$PY -m pytest tests/ppgbench -q
$PY -m ppgbench audit --dataset mendeley --protocol subject_loso
$PY -m ppgbench audit --dataset bigideas --protocol chronological --dataset-arg max_windows_per_egv=2
$PY -m ppgbench run --dataset mendeley --protocol subject_loso \
      --models mean ridge demo ridge+demo gbm gbm+demo resnet1d cnnlstm 1d 2d 3d fusion --seeds 0 1 2 --epochs 40
$PY -m ppgbench run --dataset bigideas --protocol chronological \
      --models subject_mean mean ridge 1d 2d 3d fusion --epochs 15 --batch 64 --dataset-arg max_windows_per_egv=2
$PY -m ppgbench ladder --dataset mendeley --models mean subject_mean 1d fusion --seeds 0 1 2
$PY -m ppgbench ladder --dataset bigideas --models mean subject_mean 1d --epochs 15 --batch 64 --dataset-arg max_windows_per_egv=2
$PY -m ppgbench bootstrap --dataset mendeley --protocol subject_loso
$PY -m ppgbench bootstrap --dataset bigideas --protocol chronological
$PY -m ppgbench report --latex results_macros_ppgbench.tex --markdown results/summary.md --figures figures/ppgbench
tectonic ieee_paper.tex
```

Legacy naming notes: the BIG IDEAs within-subject experiment called its
null model `mean` although it was the per-subject train mean; in the
framework that model is `subject_mean` and its macros are `\bigSubjMean*`.
The legacy ladder stored protocols as `window/record/subject` and
`reading/chrono0/chrono`; `report` maps the new names to the same tokens
(`Window/Record/Subject`, `Reading/ChronoNoGap/Chrono`), so
`\mendLadderWindowOneD` etc. keep their names.

---

## 11. Design rules for contributors

* Never evaluate on windows.  If you need a window-level diagnostic, log it;
  do not store it.
* Never early-stop, select hyper-parameters or pick seeds on the held-out
  fold.  Nested selection belongs inside `fit` / `train_net` on the training
  indices.
* Every protocol has a declared leak level and a test that proves a leaky
  split is rejected.
* Every experiment carries a null model; a model's number without the null's
  is not a result.
* Seeds everywhere; no `random` without a seed.
* `--quick` results are never reported.
* Do not edit a store by hand; rerun.  Entries are replaced by key, so a
  rerun of one model/seed updates exactly that entry.
* Add the macro to the paper, not the number.

---

## 12. Where to go next

The framework is deliberately target-agnostic: `y` is any per-record scalar
and the protocols only look at `subject`, `record` and `time`.  Registering
a new target (heart rate, blood pressure, HbA1c) is a dataset loader; the
leak discipline, nulls, bootstrap and export come for free.  Foundation
model probes (`papagei_probe` and new `Encoder`s) run under the same
protocols as the trained networks, which is what makes their comparison
meaningful.
