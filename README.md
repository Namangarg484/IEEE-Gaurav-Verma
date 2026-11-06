# A label-replication artefact, a measured leakage ladder and a null-anchored benchmark for PPG-based glucose estimation

Code, loaders, frozen protocols, integrity tests, out-of-fold predictions and
the manuscript (`ieee_paper.tex`, supplement in `supplement.tex`).

## What this repository establishes

1. **A label-replication artefact** in the public Kaggle derivative
   `clean-dataset.csv` of the Kermani–Esmaeili PPG–glucose recordings: within
   each subject the glucose label cycles row by row through the subject's
   reference values, so every window carries the subject mean and a
   per-subject oracle scores MAE 0.07 mg/dL. Subject-wise splitting cannot
   detect it; `tests/test_integrity.py::test_clean_dataset_csv_is_a_cross_join`
   streams the whole file and does. The first draft of this work trained on
   that file; its numbers are withdrawn (`legacy_ieee_paper_v1.tex`).
2. **A measured leakage ladder**: the same models under random-window,
   random-recording, subject-wise and derivative-file protocols
   (Kermani–Esmaeili) and random-reading, chronological and
   guarded-chronological protocols (BIG IDEAs).
3. **A null-anchored benchmark** over three public cohorts with null,
   classical, published-architecture (ResNet-1D, CNN-LSTM, GBM) and
   multi-view fusion rows, subject-cluster bootstrap intervals, positive
   controls, clinical anchoring (event sensitivity, reference-noise floors)
   and a fair personalised arm.

Concurrent work by Ramesh et al. (arXiv:2608.01820, ISWC '26) and
Seyedebrahimi et al. (medRxiv 10.64898/2026.08.03.26359550) established
independently that random splits inflate PPG-glucose results; this
repository provides the artefact, the ladder and the harness.

## Cohorts (all public; none redistributed here)

| Cohort | Source | Subjects / references | Reference device | Loader |
|---|---|---|---|---|
| Kermani–Esmaeili finger PPG | Mendeley Data 10.17632/37pm7jk7jn (v3 csv layout) | 23 / 67 | Accu-Chek Active | `ppg_data.load_mendeley` |
| BIG IDEAs wrist BVP + CGM | PhysioNet 10.13026/aw6y-fc44 (v1.1.3, credentialed) | 16 / 28,867 CGM readings | Dexcom G6 | `scripts/fetch_bigideas.py` → `ppg_data.load_bigideas` |
| STUBA ePPG multi-wavelength | Zenodo 10.5281/zenodo.21978226 (CC0) | 24 (3 T1DM) / 125 | Beurer GL49 | `ppg_data.load_stuba` |

Place the Mendeley archive under `Archive_Dataset/archive (2)/PPG_Dataset/`,
the BIG IDEAs zip under `data.nosync/bigideas/` and the STUBA zip extracted
under `data.nosync/stuba/wfdb_data/` (folders named `*.nosync` are excluded
from iCloud sync).

## Layout

| File | Role |
|---|---|
| `ppg_data.py` | Loaders and the common representation (0.5–8 Hz, 64 Hz, 5-s windows, z-score); `make_views` (1-D / CWT / Takens). |
| `fusion_network_model.py`, `baseline_models.py` | 1D-2D-3D cross-attention net; ResNet-1D and CNN-LSTM re-implementations. |
| `run_subject_cv.py` | Leave-one-subject-out on Mendeley or STUBA (`--dataset`), all models, recording-level metrics. |
| `run_bigideas_cv.py` | BIG IDEAs chronological split with guard gap (`within`), per-subject models (`personal`), LOSO; `--target residual` personalised arm; time-of-day and persistence nulls. |
| `run_leakage_ladder.py` | The ladder (`--dataset mendeley|bigideas|derivative`). |
| `run_controls.py` | Positive/negative controls (heart rate, age, permuted labels, white-noise input). |
| `run_preproc_ablation.py` | Window length / band / amplitude sensitivity on BIG IDEAs. |
| `run_foundation_probes.py` | Ridge probes on frozen PaPaGei-S, Pulse-PPG and AnyPPG embeddings under the same protocols (absolute and residual targets on BIG IDEAs). |
| `run_ppgbp.py` | PPG-BP (Liang et al. 2018, 219 subjects) systolic/diastolic BP under a leaky random-segment and an honest subject-wise split: shows the audit detects real signal. |
| `validate_detectors.py` | Validation on known ground truth: cross-join detector sensitivity/specificity (global vs block scan, plus real negative/positive controls) and false-claim rate / power of the null-anchored decision rule under honest vs leaky protocols. |
| `bootstrap_ci.py`, `stats_extra.py`, `clinical_metrics.py` | Subject-cluster bootstrap CIs; r CIs, Wilcoxon, sign tests, MDE, leverage; event metrics and reference-noise floors. |
| `make_dataset_stats.py`, `make_results_macros.py` | Cohort facts and every number in the paper → `results_macros.tex`. |
| `make_paper_figures.py`, `draw_architecture.py` | Figures → `figures/`. |
| `tests/test_integrity.py` | Label mapping, subject leakage, timestamp units, artefact detection. |
| `ppgbench/` | Benchmark package (named protocols with asserted no-leak guarantees, model/dataset registries, metrics, report, CLI); see `docs/framework_spec.md`, `docs/framework.md`. |
| `docs/research_synthesis.json` | Literature/venue/reviewer synthesis that drove the re-scope. |

Root-level scripts from the withdrawn pipeline (`fusion_data_pipeline.py`,
`train_fusion_model.py`, `run_ablation_baselines.py`, `run_extended_analysis.py`,
`linear_probe.py`, `analyze_bigideas_dataset.py`, `prepare_generalization_data.py`)
read the corrupted CSV and are kept only for the record.

## Reproduce

```bash
PY=~/.venvs/ppgfusion/bin/python     # Python 3.12, torch 2.15 (MPS), scipy, pywt, sklearn, pandas, wfdb

$PY -m pytest tests -q
$PY scripts/fetch_bigideas.py --zip data.nosync/bigideas/bigideas-1.1.3.zip --out data.nosync/bigideas --signals BVP

# benchmark rows
$PY run_subject_cv.py --models mean ridge demo ridge+demo gbm gbm+demo 1d 2d 3d fusion resnet1d cnnlstm --seeds 0 1 2
$PY run_subject_cv.py --folds 5 --models mean ridge 1d --seeds 0 1 2            # LOSO-bias sensitivity (kfold5_ prefix)
OMP_NUM_THREADS=1 $PY run_subject_cv.py --dataset stuba --models mean ridge gbm 1d 2d 3d fusion --seeds 0 1 2
for ch in R1 G1 R2 IR2 G2 mean; do OMP_NUM_THREADS=1 $PY run_subject_cv.py --dataset stuba --channel $ch --models mean ridge gbm; done
$PY run_bigideas_cv.py --protocol within --models mean tod persist ridge 1d 2d 3d fusion --epochs 15 --seeds 0 1 2
$PY run_bigideas_cv.py --protocol within --models mean tod persist ridge 1d fusion --target residual --epochs 15 --seeds 0 1 2

# frozen PPG foundation-model probes (downloads into data.nosync/encoders/, weights loaded weights_only=True):
#   git clone https://github.com/nokia-bell-labs/papagei-foundation-model data.nosync/encoders/papagei-foundation-model
#   git clone https://github.com/maxxu05/pulseppg data.nosync/encoders/pulseppg
#   git clone https://github.com/PKUDigitalHealth/AnyPPG data.nosync/encoders/AnyPPG      # ships anyppg_ckpt.pth
#   papagei_s.pt from Zenodo 13983110 -> data.nosync/encoders/weights/papagei/
#   pulseppg_model_weights.zip from Zenodo 17345536 -> data.nosync/encoders/weights/pulseppg/ (unzip)
$PY run_foundation_probes.py --encoders papagei anyppg pulseppg --datasets mendeley stuba bigideas   # ~2.5 h on an M4 Air

# ladder, controls, sensitivity
$PY run_leakage_ladder.py --dataset mendeley --models mean subjmean 1d fusion --seeds 0 1 2
$PY run_leakage_ladder.py --dataset bigideas --models mean subjmean 1d --seeds 0
$PY run_leakage_ladder.py --dataset derivative --models mean subjmean 1d
$PY run_controls.py --dataset mendeley --controls hr age perm noise glucose --seeds 0 1 2
$PY run_preproc_ablation.py

# statistics, clinical anchoring, paper
$PY bootstrap_ci.py && $PY stats_extra.py && $PY clinical_metrics.py && $PY make_dataset_stats.py
# PPG-BP generalisation (download the CC0 PPG-BP release from figshare into data.nosync/ppgbp/):
$PY run_ppgbp.py --models mean ridge demo ridge+demo gbm+demo          # deterministic, one seed
$PY run_ppgbp.py --models 1d --seeds 0 1 2
# validation of the audit itself (~25 min):
OMP_NUM_THREADS=1 $PY validate_detectors.py --reps-a 30 --reps 200
$PY make_results_macros.py && $PY make_paper_figures.py && $PY draw_architecture.py
$PY make_supp_tables.py
$PY scripts/order_bibliography.py tbme_paper.tex refs_tbme_pool.tex refs_tbme.tex
$PY scripts/order_bibliography.py supplement.tex refs_tbme_pool.tex refs_supp.tex
tectonic tbme_paper.tex && tectonic supplement.tex          # TBME submission
tectonic ieee_paper.tex && tectonic supplement_jbhi.tex     # earlier JBHI-style version
```

Every result in `results/` is an out-of-fold prediction file (one row per
reference measurement) or a JSON summary; the manuscript contains no typed
number (`\m{...}` renders a red dash for anything not yet computed).

## Protocol in one paragraph

Subjects never appear on both sides of a split unless the protocol is
explicitly a leaky rung of the ladder. Each fold trains a fresh model for a
fixed budget with no early stopping on the held-out subject; the target is
standardised with training statistics only. Window predictions are averaged
per reference measurement; every metric is computed on out-of-fold
measurement-level predictions next to a null model (train mean; or subject
mean, time of day and persistence for within-subject tasks), with
subject-cluster bootstrap intervals, a pre-specified primary comparison
(1-D CNN vs null) and a stated minimum detectable effect.
