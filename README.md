# iatreion

`iatreion` is an interpretable dementia differential-diagnosis pipeline based on [Rule-based Representation Learner (RRL)](https://github.com/12wang3/rrl). The current research workflow focuses on binary, pairwise diagnosis tasks such as AD versus AD-mix, AD versus FTLD, AD versus VaD, and related clinically defined groupings. It supports multimodal hospital data, internal cross-validation, final model fitting, external validation, rule inspection, and figure/table generation.

The name comes from Ancient Greek `ἰατρεῖον` ("clinic").

This README documents the RRL workflow used for the dementia manuscript, with brief instructions for XGBoost and Random Forest baseline training.

## Overview

The core pipeline is:

1. Install the package and dependencies with `uv`.
2. Convert raw hospital spreadsheets into processed `.data`, `.info`, and `process_info.toml` files.
3. Run nested internal RRL evaluation: each outer fold tunes modality-specific hyperparameters only inside its outer-training data, then evaluates the held-out outer test fold.
4. Run final RRL tuning on all internal data with one-layer CV, fit the final modality-specific models on all internal data, and write the final fusion artifact.
5. Validate final models on external data with the parser.
6. Generate tables, ROC plots, rule waterfalls, or use the GUI.

Most commands expose detailed help:

```bash
uv run iatreion -h
uv run iatreion process -h
uv run iatreion train -h
uv run iatreion train rrl -h
uv run iatreion train xgboost -h
uv run iatreion train random-forest -h
uv run iatreion train rrl-eval -h
uv run iatreion rrl-eval -h
uv run iatreion show -h
uv run iatreion show rrl-waterfall -h
```

## Installation

Clone the repository:

```bash
git clone https://github.com/PumchProjects/iatreion.git && cd iatreion
```

The project is managed by `uv` and requires Python 3.12 (which will be installed by `uv`). Choose the PyTorch dependency that matches your machine:

```bash
uv sync --extra cpu    # No CUDA
uv sync --extra cu121  # CUDA 12.1
uv sync --extra cu124  # CUDA 12.4
uv sync --extra cu126  # CUDA 12.6
```

The main entry points are:

```bash
uv run iatreion      # CLI
uv run iatreion-gui  # GUI
```

## Configuration Files

Most commands can read defaults from `configs/config.toml` with the global `--config` option. The file is organized by command: `[process]` and `[process.data]` for raw-to-processed conversion, `[train.rrl]`, `[train.rrl-eval]`, `[train.xgboost]`, and `[train.random-forest]` for model runs, and `[show.*]` tables for figure/table helpers.

CLI options override TOML values, so `uv run iatreion train rrl --config configs/config.toml -i 6-7` uses the config but overrides `train.rrl.device-id`. Other commands that accept `--config` behave the same way, for example `uv run iatreion process --config configs/config.toml` or `uv run iatreion show table-1 --config configs/config.toml -o table_1_retry`.

Before using `configs/config.toml`, replace its placeholder paths with real local paths, including values such as `process.prefix`, `process.group-data`, `process.basic-data`, `process.data.*`, `process.vmri`, `process.vmri-change`, `train.rrl.prefix`, `train.xgboost.prefix`, `train.random-forest.prefix`, and show command `prefix` values. You can also leave the TOML generic and pass real paths from the CLI; command-line values take precedence.

RRL hyperparameter tuning uses a separate TOML search space, usually `configs/rrl_optuna.toml`, selected with `--tune-config` or `train.rrl.tune-config`. Its `[execution]` table controls Optuna execution settings such as `trial-log-root`, while `configs/config.toml` or CLI options control the training log root.

## Data Model

Processed datasets are selected with `-n/--names` or the relevant `names` entry in `configs/config.toml`. In the current RRL manuscript workflow, the main modalities are:

| Dataset name | Source data | Role |
| --- | --- | --- |
| `symptom` | Medical history spreadsheet | Symptom and history-derived variables |
| `s-screen-sum` | Cognitive screening spreadsheet | Demographics plus MMSE/MoCA/ADL/HAD summary domains |
| `composite-bin` | Cognitive composite spreadsheet | Composite cognitive test features and binarized domains |
| `biomarker` | Blood biomarker spreadsheet | Aβ42, ptau217, GFAP, NFL, and ratios over Aβ42 |
| `cbf` | MRI CBF spreadsheet | Cerebral blood-flow features |
| `csvd` | MRI CSVD spreadsheet | Cerebral small vessel disease features |
| `volume-new-pct` | MRI volume spreadsheet | Age-normalized MRI volume percentage features |

Several dataset names are composites. For example, `s-screen-sum` is built from `basic`, `mmse-sum`, `moca-sum`, `adl-sum`, and `had-sum`.

Raw source paths are provided at preprocessing time with `--data.<raw-data-name>` or `[process.data]` entries in `configs/config.toml`, where raw data names include `history`, `screen`, `composite`, `biomarker`, `cbf`, `csvd`, and `volume-new`. The input data are not distributed with this repository. Default date/index column rules are defined in `src/iatreion/configs/preprocessor.py`.

## Labels and Groups

Groups are selected with `-g/--groups`. Each `-g` argument is one class. A group string can merge several encrypted subgroups; for example `-g ac f` means `AD + AD-mix` versus healthy controls.

Common display mappings used by plotting helpers include:

| Code | Display name |
| --- | --- |
| `a` | AD |
| `c` | AD-mix |
| `ac` | AD + AD-mix |
| `dgn` | FTLD |
| `o` | VaD |
| `f` | HC |
| `1` | Aβ+ |
| `2` | Aβ- |

The default label column in processed internal data is `group_encrypted`. For external validation, use `--label-name` to set the label column.

## Preprocessing

Run preprocessing once for the internal hospital data:

```bash
uv run iatreion process \
  -p "<path-to-a-new-or-non-existent-folder>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --group-data "<path-to-the-patient-group-mapping-file>" \
  --basic-data "<path-to-the-basic-patient-information-file>" \
  --data.history "<path-to-the-history-spreadsheet>" \
  --data.screen "<path-to-the-cognitive-screening-spreadsheet>" \
  --data.composite "<path-to-the-cognitive-composite-spreadsheet>" \
  --data.biomarker "<path-to-the-blood-biomarker-spreadsheet>" \
  --data.cbf "<path-to-the-mri-cbf-spreadsheet>" \
  --data.csvd "<path-to-the-mri-csvd-spreadsheet>" \
  --data.volume-new "<path-to-the-mri-volume-spreadsheet>" \
  -v "<path-to-the-mri-volume-mean-std-file>" \
  --vmri-change "<path-to-the-mri-volume-column-change-file>"
```

Or after replacing placeholder paths with real local paths:

```bash
uv run iatreion process --config "configs/config.toml"
```

This creates, for each selected dataset:

| Output | Meaning |
| --- | --- |
| `<name>.data` | Processed sample-by-feature table with label columns |
| `<name>.info` | Feature metadata: index, feature type, category labels, label columns |
| `process_info.toml` | Persistent preprocessing metadata needed for external validation |

Important raw preprocessing behavior:

- If a raw Excel workbook needs a non-default sheet, pass `--data-sheets.<raw-data-name> <sheet-name-or-index>`.
- Discrete variables are stored as category codes with category metadata in `.info` and `process_info.toml`.
- Raw preprocessing does not perform missingness-based sample filtering, manuscript-grade feature selection, imputation, normalization, under-sampling, or final model encoding; those steps are fitted later inside each training fold.

## Aggregation Modes

`-a/--aggregate` controls how multiple modalities are handled.

| Mode | Meaning | Typical use |
| --- | --- | --- |
| `concat` | Merge all selected modality features into one table and train one model | Hyperparameter tuning without fusion bootstrapping |
| `average` | Train/evaluate one model per modality and average available probabilities | Simple baseline |
| `calibrated-concat` | Merge all selected modality features into one model, then calibrate its logit and tune the clinical threshold | Concatenated-feature model with calibrated operating point |
| `calibrated-fusion` | Train one model per modality, calibrate each modality logit, then combine available modalities with equal weights | Main multimodal workflow |

`calibrated-fusion` first trains one model per modality. For each modality, the positive-class probability is transformed to a logit and calibrated by a one-dimensional logistic regression on inner-fold predictions. At inference time, available calibrated modality logits are combined with equal modality weights; missing modalities are omitted and the remaining weights are renormalized.

`calibrated-concat` first concatenates features from all selected modalities into one model, then applies the same one-dimensional logit calibration and clinical-threshold selection to that single concatenated model. It is useful when the desired comparison is a single early-fusion model rather than per-modality late fusion.

For binary tasks, `calibrated-concat` and `calibrated-fusion` both store a clinical threshold in `available_fusion.toml`. The threshold is chosen to satisfy a target recall for a specified class, controlled by `--clinical-threshold-label` and `--clinical-threshold-recall`.

## Fold-fitted Training Preprocessing

Training commands read the processed `.data` and `.info` files, then fit preprocessing artifacts inside each training fold. This keeps validation, outer-test, and external data out of fold-specific feature selection, imputation statistics, normalization statistics, and under-sampling decisions.

The main fold-fitted preprocessing controls are:

| Option | Meaning |
| --- | --- |
| `-k/--keep` | Resolve duplicate samples during training and parser evaluation; the default is `last` |
| `--feature-selection-method/--feature-selection-fraction/--feature-selection-top-k/--feature-selection-c` | Supervised feature selection method, kept fraction, exact top-k, or L1-logistic inverse regularization; `--feature-selection-method none` disables it |
| `--feature-selection-min-features/--feature-selection-max-features/--feature-selection-score-aggregate` | Minimum/maximum selected raw features and category-score aggregation (`max` or `mean`) |
| `--missing-value-strategy` | Fold-fitted missing-value handling: `simple`, `limix`, or `none` |
| `--no-normalize-continuous` | Disable continuous-feature z-scoring |
| `--under-sampler` | Apply optional under-sampling to each training fold; currently `random` |
| `--target-n-samples` | Maximum samples kept per class after under-sampling; `0` balances to the smallest class |

Optional supervised feature selection is applied before imputation, normalization, under-sampling, and final encoding. Available methods are:

| Method | Meaning |
| --- | --- |
| `none` | Keep all raw features |
| `f_classif` | ANOVA F-test; supports binary and multiclass labels |
| `mutual_info` | Mutual-information classifier score; supports binary and multiclass labels |
| `l1_logistic` | Multinomial L1-regularized logistic-regression embedded selection |
| `auc` | Binary-only univariate AUROC distance from chance |
| `logistic_lrt` | Binary-only univariate logistic likelihood-ratio score |

Unordered categorical variables are scored through temporary one-hot columns and aggregated back to raw features. `l1_logistic` uses tunable inverse regularization strength, `auc` ranks features by distance from chance AUROC, and `logistic_lrt` ranks each raw feature by logistic deviance improvement over an intercept-only model.

Continuous features are z-scored during model training using training-fold statistics unless `--no-normalize-continuous` or `--no-pp` is used. Optional random under-sampling is applied to training folds only after imputation and continuous-feature normalization, and before feature encoding; validation and test folds are not under-sampled.

## RRL Training Concepts

RRL learns a non-fuzzy rule representation and exports readable rules as TSV files. The important RRL options are:

| Option | Meaning |
| --- | --- |
| `-s/--structure` | Binarization and logical layer sizes, for example `5@256` |
| `--use-not` | Enable NOT terms in rules |
| `--skip` | Enable skip connections between logical layers |
| `--nlaf --alpha --beta --gamma` | Use novel logical activation functions and their parameters |
| `--temp` | Initial softmax temperature |
| `--weighted` | Use class-balanced cross-entropy weights |
| `--val-size` | Reserve a validation split inside each training fold |
| `--early-stop-patience/--early-stop-min-delta` | Early stopping patience and minimum validation F1 improvement |
| `--label-smoothing` | Label smoothing |
| `--max-grad-norm` | Gradient clipping norm |
| `-v improved --tau --kappa` | Missing-aware RRL with coverage-gated logic |

For missing-aware RRL (`-v improved`), missing values are intentionally kept during RRL training and the model receives both values and observation masks.

## Hyperparameter Tuning

RRL hyperparameter search uses Optuna and a TOML search space. The default search space is `configs/rrl_optuna.toml`; its `study.objective` is now a metric name such as `AUROC`, `AUPRC`, or `F1`, and the runner resolves it to the current tuning target, for example `symptom/AUROC` or `all_concat/AUROC`. TOML keys use kebab-case, so the search space can include `train.feature-selection.*` entries; the runner maps those keys back to Python attribute names before applying overrides. Supervised feature-selection method and kept-feature fraction are tuned inside the same nested CV loop as the RRL hyperparameters; for multiclass tasks, exclude binary-only choices such as `auc` and `logistic_lrt` from the search space.

Passing `--tune-config` selects the manuscript-grade workflow. Without `-f`, the command runs nested internal evaluation: for every outer fold, each modality is tuned on inner CV splits from the outer-training data only, then the selected per-modality parameters are used to train/calibrate on the outer-training data and evaluate the held-out outer test fold. With `-f`, the command runs final training: each modality is tuned once on all internal data with one-layer CV, an internal OOF fusion artifact is fitted, and final models are fitted on all internal data.

Use `-i` to expose CUDA devices. A single device can be written as `-i 0`; multiple devices can be written as `-i 0,1` or `-i 0-7`. Each SQLite-backed Optuna study runs one trial at a time (`execution.n-jobs = 1`) to avoid database-lock failures during `study.ask()`/`tell()`; crashed `RUNNING` trials are marked failed on resume, and already completed trials are reused instead of rerun.

During tuning, terminal output is intentionally compact: it reports study start, one-line trial summaries, and the current best value. Detailed fold, epoch, calibration, and metric logs are written to each `trial_<n>/trial_<n>.log`, while `study.log`, `trial_info.toml`, `best_trial.toml`, and `params.toml` summarize the Optuna study artifacts.

## Nested Internal RRL Evaluation

Run the full multimodal nested evaluation directly; there is no manual parameter-copying step:

```bash
uv run iatreion train rrl \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  -g a c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --log-root logs \
  -i 0-7 \
  --val-size 0.2 \
  -e 1201 --batch-size 128 --save-interval 100 \
  --early-stop-patience 10 --early-stop-min-delta 0.001 --label-smoothing 0.05 --max-grad-norm 3.0 \
  --nlaf --alpha 0.9 --beta 3 --gamma 3 \
  --use-not --skip --weighted \
  -v improved --missing-value-strategy none \
  --tune-config configs/rrl_optuna.toml
```

The training output root is `logs` in this example; set it with `--log-root` or `train.rrl.log-root` in `configs/config.toml`. The Optuna artifact root shown below defaults to `logs_optuna_rrl`, controlled by `execution.trial-log-root` in `configs/rrl_optuna.toml`.

Optuna artifacts are organized by stage, outer fold, and tuning target:

```text
logs_optuna_rrl/<study-name>/nested/outer_<k>/<target>/study.db
logs_optuna_rrl/<study-name>/nested/outer_<k>/<target>/best_trial.toml
logs_optuna_rrl/<study-name>/nested/selected_params.toml
```

Nested evaluation output is organized under:

```text
logs/<dataset-names>/<group-names>/rrl/calibrated-fusion/
```

The important exported files include:

| File pattern | Meaning |
| --- | --- |
| `train.log` | Training log |
| `rrl_<name>_<outer>_<inner>.tsv` | Exported RRL rule table for one modality/fold |
| `rrl_<name>_<outer>_<inner>.feature-selection.toml` | Fold-fitted supervised feature-selection artifact when feature selection is enabled |
| `train_avg_<result>.log` | Mean/std fold metrics |
| `train_ci_<result>.log` | Bootstrap confidence intervals |
| `results_<result>.npz` | Saved `y_true`, `y_pred`, `y_score`, `y_mask`, and fold metrics |
| `roc_<result>.png` | ROC curve with AUROC |

For `calibrated-fusion`, aggregate result names are:

```text
all_calibrated_fusion_original
all_calibrated_fusion_clinical_recall
```

For `calibrated-concat`, aggregate result names are:

```text
all_calibrated_concat_original
all_calibrated_concat_clinical_recall
```

## Parser Re-evaluation

Downstream tools use the RRL parser rather than the live PyTorch model. Parser-based internal re-evaluation is optional for checking exported rule files against the live-model nested CV output:

```bash
uv run iatreion train rrl-eval \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  -g a c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --no-pp \
  --log-root logs
```

Use `--no-pp` when the exported rules should be evaluated directly on processed feature values. In the current missing-aware RRL workflow, this is the usual choice for parser re-evaluation and external validation. Supervised feature selection is not re-fitted during parser evaluation; final rules access the selected features by column name.

Parser re-evaluation does not train a live RRL model and does not use a validation split, so `--val-size` belongs in live RRL training settings rather than `[train.rrl-eval]`.

This command writes parser-based metrics under:

```text
logs/<dataset-names>/<group-names>/rrl-discrete/calibrated-fusion/
```

To evaluate or publish a modality subset while keeping the split determined by all input modalities, use `--eval-names`:

```bash
uv run iatreion train rrl-eval \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --eval-names symptom csvd \
  -g a c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --no-pp \
  --log-root logs
```

In this mode, split files still come from all listed modalities, but fusion, calibration, thresholding, and metrics only use `symptom` and `csvd`. Samples missing both selected modalities are excluded from metrics.

## Final Model Fitting

For external validation, run final tuning and final fitting on all internal data with `--tune-config -f`. The final command should match the internal nested-evaluation settings, including any `--eval-names` subset.

```bash
uv run iatreion train rrl \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  -g a c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --log-root logs \
  -i 0-7 \
  --val-size 0.2 \
  -e 1201 --batch-size 128 --save-interval 100 \
  --early-stop-patience 10 --early-stop-min-delta 0.001 --label-smoothing 0.05 --max-grad-norm 3.0 \
  --nlaf --alpha 0.9 --beta 3 --gamma 3 \
  --use-not --skip --weighted \
  -v improved --missing-value-strategy none \
  --tune-config configs/rrl_optuna.toml \
  -f
```

Final tuning artifacts use the same `execution.trial-log-root` setting from `configs/rrl_optuna.toml`; final model and rule artifacts use `--log-root` or `train.rrl.log-root`.

Final tuning artifacts are written under:

```text
logs_optuna_rrl/<study-name>/final/<target>/study.db
logs_optuna_rrl/<study-name>/final/<target>/best_trial.toml
logs_optuna_rrl/<study-name>/final/selected_params.toml
```

The final runner fits an internal OOF `available_fusion.toml` with the selected final parameters and publishes it to:

```text
logs/final/<group-names>/rrl/available_fusion.toml
```

Final rule files are saved as:

```text
logs/final/<group-names>/rrl/<name>.tsv
```

## External Validation

Use `uv run iatreion rrl-eval` to apply final RRL rule files to external data. The command uses:

- `-t/--thesaurus`: root directory containing final RRL logs.
- `-p/--process`: internal `process_info.toml`, needed to reproduce feature encodings and category orders.
- `--data.<raw-data-name>`: external spreadsheet for each requested raw data source.
- `--data-sheets.<raw-data-name>`: optional Excel sheet name or index for a raw data source.
- `--index-name`: external sample ID column.
- `--label-name`: external label column, required for `-m eval`.

Example for symptom plus CSVD:

```bash
uv run iatreion rrl-eval \
  -n symptom csvd \
  -g a c \
  -t logs \
  -p "<path-to-the-process-info-file>" \
  --data.history "<path-to-the-spreadsheet-for-symptom-data>" \
  --data.csvd "<path-to-the-spreadsheet-for-csvd-data>" \
  -m eval \
  -k last \
  --index-name "<sample-id-column>" \
  --label-name "<label-column>" \
  -D
```

Example including MRI volume features:

```bash
uv run iatreion rrl-eval \
  -n symptom csvd volume-new-pct \
  -g a c \
  -t logs \
  -p "<path-to-the-process-info-file>" \
  --data.history "<path-to-the-spreadsheet-for-symptom-data>" \
  --data.csvd "<path-to-the-spreadsheet-for-csvd-data>" \
  --data.volume-new "<path-to-the-spreadsheet-for-mri-volume-data>" \
  -v "<path-to-the-file-storing-mean-and-std-for-mri-volume-data>" \
  --vmri-change "<path-to-the-file-storing-column-name-changes-for-mri-volume-data>" \
  -m eval \
  -k last \
  --index-name "<sample-id-column>" \
  --label-name "<label-column>" \
  -D
```

Evaluation modes:

| Mode | Meaning |
| --- | --- |
| `single` | Show prediction and active supporting/opposing rules for one sample |
| `batch` | Produce predictions for a batch without metrics |
| `eval` | Compute metrics when external labels are available |
| `show` | List exported model rules |

For single-sample explanations, pass `--sample-id`.

## GUI

After final model fitting, launch:

```bash
uv run iatreion-gui
```

The GUI wraps the same parser API used by `iatreion rrl-eval`. It can load final RRL models, select input files, run batch predictions, view active rules, and evaluate labeled external data.

## XGBoost and Random Forest Baselines

XGBoost and Random Forest can be trained with the same processed `.data` and `.info` files as RRL. Their defaults can live in `[train.xgboost]` and `[train.random-forest]` in `configs/config.toml`; common fields such as `prefix`, `names`, `groups`, `aggregate`, `clinical-threshold-label`, `clinical-threshold-recall`, and `log-root` follow the same config/CLI override rules described above.

With the config file, run:

```bash
uv run iatreion train xgboost --config configs/config.toml
uv run iatreion train random-forest --config configs/config.toml
```

CLI arguments override the TOML values, so you can keep the common baseline setup in the file and change only the machine- or run-specific values:

```bash
uv run iatreion train xgboost --config configs/config.toml -i 6-7 --device cuda
uv run iatreion train random-forest --config configs/config.toml -p "<path-to-the-folder-storing-processed-data>"
```

The equivalent direct commands are:

```bash
uv run iatreion train xgboost \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom csvd \
  -g a c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --log-root logs \
  -i 0-7 \
  --device cuda

uv run iatreion train random-forest \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom csvd \
  -g a c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --log-root logs
```

The baseline outputs follow the same dataset/group/aggregate directory convention as other training commands, with model-specific folders such as `xgboost` and `random_forest` under the selected log root.

## Figures and Tables

Use `uv run iatreion show` for manuscript tables and figures.

Common outputs:

```bash
uv run iatreion show table-1 \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  -g a c \
  -o table_1
```

```bash
uv run iatreion show latex-ci-delong \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  -g a c \
  -m rrl \
  -a calibrated-fusion \
  -r all_calibrated_fusion_clinical_recall \
  -l RRL-Calibrated-Fusion \
  -o table_ci_delong
```

```bash
uv run iatreion show rrl-waterfall \
  -n symptom csvd \
  -g a c \
  -t logs \
  -p "<path-to-the-process-info-file>" \
  --data.history "<path-to-the-spreadsheet-for-symptom-data>" \
  --data.csvd "<path-to-the-spreadsheet-for-csvd-data>" \
  -k last \
  --index-name "<sample-id-column>" \
  --sample-id "<sample-id>" \
  --top-k 20 \
  --title "RRL Waterfall Plot" \
  -o sample_rrl_waterfall
```

Generated figures and tables are written to `figures/` by default.

## Metrics and Statistical Tests

Training and evaluation report:

- AUROC
- AUPRC
- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Sensitivity and specificity for binary tasks
- Confusion matrix
- Training time
- RRL complexity as `Log#E`, the log number of rule edges

Confidence intervals use bootstrap resampling. Defaults:

```text
--bootstrap-samples 1000
--ci-level 0.95
```

Plot/table helpers include:

- Wilcoxon signed-rank tests over fold metrics.
- DeLong tests over full out-of-fold AUROCs.
- McNemar tests for paired accuracy comparisons.

For binary labels, confirm which class is treated as the positive class in each analysis. Internally, class order follows the sorted selected group labels; AUROC and AUPRC use the second sorted class as the positive class, and AUPRC is reported as sklearn average precision rather than interpolated trapezoidal PR-AUC.

## Repository Layout

```text
configs/
  config.toml                      # Main CLI defaults
  rrl_optuna.toml                  # Optuna study/search-space defaults
src/iatreion/
  cli/                             # CLI commands
  configs/                         # CLI dataclasses and config semantics
  preprocessors/                   # Raw-to-processed feature extraction
  train_utils/                     # Splitting, feature selection, encoding, fusion, imputation artifacts
  trainers/                        # Training loop, metric recording, final artifacts
  models/rrl.py                    # Live RRL training wrapper
  models/rrl_discrete.py           # Parser/evaluator for exported RRL rules
  rrl/                             # RRL network and rule export implementation
  show_helpers/                    # Manuscript figure/table helpers
  gui/                             # Tk GUI for final RRL evaluation
```

## Troubleshooting

### Available-fusion artifact not found

Run final training with `uv run iatreion train rrl ... --tune-config configs/rrl_optuna.toml -f` before external validation. If using `--eval-names`, use the same `--eval-names` list for nested evaluation, final training, and external validation.

### No experiment root found

Check that `-p`, `-n`, `-g`, `-a`, `--log-root`, and `--eval-names` match the previous training command.

### External validation cannot find a column

Confirm the external index column with `--index-name`, label column with `--label-name`, and raw data mapping such as `--data.history ...`, `--data.csvd ...`, or `--data.volume-new ...`. MRI volume validation also needs `-v` and `--vmri-change`.

### Parser metrics use fewer samples than expected

Samples with all selected modalities missing are masked out. If using `--eval-names symptom csvd`, any sample missing both symptom and CSVD is excluded from metric calculation.
