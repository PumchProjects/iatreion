# iatreion

`iatreion` is an interpretable dementia differential-diagnosis pipeline based on [Rule-based Representation Learner (RRL)](https://github.com/12wang3/rrl). The current research workflow focuses on binary, pairwise diagnosis tasks such as AD versus AD-mix, AD versus FTLD, AD versus VaD, and related clinically defined groupings. It supports multimodal hospital data, internal cross-validation, final model fitting, external validation, rule inspection, and figure/table generation.

The name comes from Ancient Greek `ἰατρεῖον` ("clinic").

This README documents the workflow used for the dementia manuscript, including the RRL model, XGBoost and Random Forest baselines, external validation, and figure/table helpers.

## Overview

The manuscript results can be reproduced from `configs/config.toml` with the command sequence in `scripts/pipeline.sh`. The rest of this README explains the same workflow step by step for inspection, debugging, and custom runs.

The core workflow is:

1. Install the package and dependencies with `uv`.
2. Convert raw hospital spreadsheets into processed `.data`, `.info`, and `process_info.toml` files.
3. Run nested internal evaluation for RRL and baseline models: each outer fold tunes modality-specific hyperparameters only inside its outer-training data, then evaluates the held-out outer test fold.
4. Run final tuning on all internal data with one-layer CV, fit the final modality-specific models on all internal data, and write final fusion artifacts.
5. Publish modality-subset fusion artifacts with result replay.
6. Validate final models on external data.
7. Generate tables, ROC plots, rule waterfalls, or use the GUI.

Most commands expose detailed help:

```bash
uv run iatreion -h
uv run iatreion process -h
uv run iatreion train -h
uv run iatreion train rrl -h
uv run iatreion train rrl-parser -h
uv run iatreion train xgboost -h
uv run iatreion train random-forest -h
uv run iatreion train result-replay -h
uv run iatreion eval -h
uv run iatreion eval rrl -h
uv run iatreion eval xgboost -h
uv run iatreion eval random-forest -h
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

## Manuscript Reproduction

Set the path variables at the top of `scripts/pipeline.sh` for the processed-data folder, the internal harmonized spreadsheet used by `process`, and the external harmonized spreadsheet used by `eval`. The script derives `process_info.toml` from the processed-data folder and passes these paths on the command line, overriding the placeholder path values in `configs/config.toml`.

`configs/config.toml` still provides the reusable preprocessing settings, modality names, model defaults, tuning files, and external-validation defaults for the manuscript run. Then execute:

```bash
bash scripts/pipeline.sh
```

The script first runs `process` and then runs three manuscript label tasks: `A` (`A+` versus `A-`), `T` (`T+` versus `T-`), and `MMSE_progression_group` (`fast` versus `slow`).

To run the paired experiment with the internal and external harmonized spreadsheets swapped, set `swap_harmonized=true` at the top of `scripts/pipeline.sh`.

For each task, the script runs two missing-data workflows. The `logs_imputed` workflow uses the default imputation settings; XGBoost and Random Forest also use native feature importance. The `logs_not_imputed` workflow trains XGBoost and Random Forest with `--missing-value-strategy none`, and trains RRL with `--missing-aware-mode improved`.

For each model/workflow combination, it runs nested internal evaluation, final fitting, full-modality result replay, two manuscript modality-subset result replays, final subset-artifact publication, and external validation for the two subset definitions:

```text
h-demo h-mmse h-moca h-mri h-history sh-apoe-labdata
h-demo h-mmse h-moca h-mri-roi h-history sh-apoe-labdata
```

It also runs `train rrl-parser` once per task and missing-data workflow as a parser parity/internal re-evaluation check. Figure and table commands are documented in the Figures and Tables section; `show` commands will be appended to `scripts/pipeline.sh` when the manuscript plotting workflow is finalized.

## Configuration Files

Most commands can read defaults from `configs/config.toml` with the global `--config` option. The file is organized by command: `[process]` and `[process.data]` for raw-to-processed conversion, `[train.rrl]`, `[train.rrl-parser]`, `[train.result-replay]`, `[train.xgboost]`, and `[train.random-forest]` for model runs, `[eval.rrl]`, `[eval.xgboost]`, and `[eval.random-forest]` for final-model external validation, and `[show.*]` tables for figure/table helpers.

CLI options override TOML values, so `uv run iatreion train rrl --config configs/config.toml -i 6-7` uses the config but overrides `train.rrl.device-id`. Other commands that accept `--config` behave the same way, for example `uv run iatreion process --config configs/config.toml` or `uv run iatreion show table-1 --config configs/config.toml -o table_1_retry`.

Before using `configs/config.toml`, replace its placeholder paths with real local paths, including values such as `process.prefix`, `process.group-data`, `process.basic-data`, `process.data.*`, `process.vmri`, `process.vmri-change`, `train.rrl.prefix`, `train.xgboost.prefix`, `train.random-forest.prefix`, and show command `prefix` values. You can also leave the TOML generic and pass real paths from the CLI; command-line values take precedence.

Hyperparameter tuning uses separate TOML search spaces selected with `--tune-config` or the model-specific `train.<model>.tune-config`: RRL usually uses `configs/optuna_rrl.toml`, XGBoost uses `configs/optuna_xgboost.toml`, and Random Forest uses `configs/optuna_random_forest.toml`. Each tuning file's `[execution]` table controls Optuna execution settings such as failure values and worker counts, while `configs/config.toml` or CLI options control the shared log root; tuning studies are always written under `{log-root}/optuna`.

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

Raw source paths are provided at preprocessing time with `--data.<raw-data-name>` or `[process.data]` entries in `configs/config.toml`, where raw data names include `history`, `screen`, `composite`, `biomarker`, `cbf`, `csvd`, and `volume-new`; raw source spreadsheets may be Excel, CSV, or TSV. The fixed MRI volume mean/std file used by `-v/--vmri` remains an Excel workbook with `mean` and `sd` sheets. The input data are not distributed with this repository. Date column rules are defined in `src/iatreion/configs/preprocessor.py`, and preprocessing receives the sample ID column from `--index-name`.

## Labels and Groups

Groups are selected with `-g/--groups`. Each `-g` argument is one class. Bare group names match labels exactly, while a leading `@` merges encrypted subgroups; for example `-g @ac f` means `AD + AD-mix` versus healthy controls. Ranges are supported inside encrypted groups, so `@a-c` is equivalent to `@abc`. Binary tasks must also set `--positive-label`; the selected positive label is encoded internally as class index `1`, so AUROC, AUPRC, ROC plots, calibrated fusion logits, SHAP's default binary output, and operating thresholds all refer to that label.

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

Preprocessing must set `--index-name` for the sample ID column and `--group-columns` to mark all label columns saved in processed data, and training commands that read processed data must set `--label-name` to choose one of those label columns.

## Preprocessing

Run preprocessing once for the internal hospital data:

```bash
uv run iatreion process \
  -p "<path-to-a-new-or-non-existent-folder>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --index-name "<sample-id-column>" \
  --group-data "<path-to-the-patient-group-mapping-file>" \
  --group-columns group_encrypted group_Ab "AC to 3" "AC 60" \
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

- If a raw Excel workbook needs a non-default sheet, pass `--data-sheets.<raw-data-name> <sheet-name-or-index>`; CSV and TSV inputs are single-table files.
- Discrete variables are stored as category codes with category metadata in `.info` and `process_info.toml`.
- Raw preprocessing does not perform missingness-based sample filtering, manuscript-grade feature selection, imputation, normalization, under-sampling, or final model encoding; those steps are fitted later inside each training fold.

## Aggregation Modes

`-a/--aggregate` controls how multiple modalities are handled.

| Mode | Meaning | Typical use |
| --- | --- | --- |
| `concat` | Merge all selected modality features into one table and train one model | Hyperparameter tuning without fusion bootstrapping |
| `average` | Train/evaluate one model per modality and average available probabilities | Simple baseline |
| `calibrated-concat` | Merge all selected modality features into one model, then calibrate its logit and tune operating thresholds | Concatenated-feature model with calibrated operating points |
| `calibrated-fusion` | Train one model per modality, calibrate each modality logit, then combine available modalities with learned non-negative weights | Main multimodal workflow |

`calibrated-fusion` first trains one model per modality. For each modality, the positive-class probability is transformed to a logit and calibrated by a one-dimensional logistic regression on inner-fold predictions. The calibrated inner-fold logits then fit one global set of non-negative modality weights by minimizing binary log loss on the simplex. At inference time, missing modalities are omitted and the remaining learned weights are renormalized; availability masks are not used as prediction features. If all learned weights for the available modalities are zero, the available calibrated logits are averaged as a fallback.

`calibrated-concat` first concatenates features from all selected modalities into one model, then applies the same one-dimensional logit calibration and operating-threshold selection to that single concatenated model. It is useful when the desired comparison is a single early-fusion model rather than per-modality late fusion.

For binary tasks, `calibrated-concat` and `calibrated-fusion` store operating thresholds in `available_fusion.toml`: a Youden-index threshold is always fitted, and a clinical recall threshold is fitted by default to satisfy a target recall for a specified class, controlled by `--clinical-threshold-label` and `--clinical-threshold-recall`; if no clinical threshold label is provided, it defaults to `--positive-label`. Pass `--no-clinical-threshold` to skip the clinical label/recall threshold; final parser evaluation and single-sample prediction then use the Youden threshold as the default operating point.

## Fold-fitted Training Preprocessing

Training commands read the processed `.data` and `.info` files, then fit preprocessing artifacts inside each training fold. This keeps validation, outer-test, and external data out of fold-specific feature selection, imputation statistics, normalization statistics, and under-sampling decisions.

The main fold-fitted preprocessing controls are:

| Option | Meaning |
| --- | --- |
| `-k/--keep` | Resolve duplicate samples during training and parser evaluation; the default is `last` |
| `--feature-selection-method/--feature-selection-fraction/--feature-selection-top-k/--feature-selection-c` | Supervised feature selection method, kept fraction, exact top-k, or L1-logistic inverse regularization; `--feature-selection-method none` disables it |
| `--feature-selection-min-features/--feature-selection-max-features/--feature-selection-score-aggregate` | Minimum/maximum selected raw features and category-score aggregation (`max` or `mean`) |
| `--no-missingness-filter/--missingness-filter-max-missing-rate/--missingness-filter-min-observed` | Fold-fitted raw-feature missingness filter; by default, features with >95% missing values or fewer than 10 observed training-fold values are removed before feature selection and imputation |
| `--missing-value-strategy` | Fold-fitted missing-value handling: `simple`, `limix`, or `none` |
| `--no-normalize-continuous` | Disable continuous-feature z-scoring |
| `--under-sampler` | Apply optional under-sampling to each training fold; currently `random` |
| `--target-n-samples` | Maximum samples kept per class after under-sampling; `0` balances to the smallest class |

The missingness filter is applied before optional supervised feature selection, and both are fitted only on the current training fold; removed features are excluded from fold rules, simple-imputer statistics, baseline transform artifacts, and modality-availability masks. Available feature-selection methods are:

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

Hyperparameter search uses Optuna and a TOML search space. `study.objective` is a metric name such as `AUROC`, `AUPRC`, or `F1`, and the runner resolves it to the current tuning target, for example `symptom/AUROC` or `all_concat/AUROC`. TOML keys use kebab-case and are mapped back to Python attribute names before applying overrides, so `learning-rate` tunes `learning_rate` and nested keys such as `train.feature-selection.method` remain possible.

Passing `--tune-config` selects the manuscript-grade workflow. Without `-f`, the command runs nested internal evaluation: for every outer fold, each modality is tuned on inner CV splits from the outer-training data only, then the selected per-modality parameters are used to train/calibrate on the outer-training data and evaluate the held-out outer test fold. With `-f`, the command runs final training: each modality is tuned once on all internal data with one-layer CV, an internal OOF fusion artifact is fitted, and final models are fitted on all internal data.

Use `-i` to expose CUDA devices. A single device can be written as `-i 0`; multiple devices can be written as `-i 0,1` or `-i 0-7`. Each SQLite-backed Optuna study runs one trial at a time (`execution.n-jobs = 1`) to avoid database-lock failures during `study.ask()`/`tell()`, while independent modality studies run in separate worker processes across the exposed devices; omit `execution.study-workers` to use one worker per exposed device, or set it to cap the number of simultaneous modality studies. On resume, unfinished trials left `RUNNING` by an interrupted process are marked failed; completed trials are reused, and new trials continue until the requested number of `COMPLETE` trials is reached.

During tuning, terminal output is intentionally compact: it reports study start, one-line trial summaries, and the current best value. Detailed fold, epoch, calibration, and metric logs are written to each `trial_<n>/trial_<n>.log`, while `study.log`, `trial_info.toml`, `best_trial.toml`, and `params.toml` summarize the Optuna study artifacts.

## Nested Internal RRL Evaluation

Run the full multimodal nested evaluation directly; there is no manual parameter-copying step:

```bash
uv run iatreion train rrl \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
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
  --tune-config configs/optuna_rrl.toml
```

The training output root is `logs` in this example; set it with `--log-root` or `train.rrl.log-root` in `configs/config.toml`. Optuna artifacts are written under `{log-root}/optuna/<study-name>/<model>/...`, so RRL and baseline studies for the same dataset/group setup share one experiment namespace without sharing SQLite databases. Internal nested-evaluation outputs are written under `{log-root}/training/<dataset-names>/<group-names>/...`; final model artifacts remain under `{log-root}/final/...`.

Every training run writes a compact `manifest.toml` in its output directory. The manifest records the command line, git commit and dirty files, `uv.lock` version/revision/hash, Python/uv/PyTorch/CUDA/GPU environment, processed data hashes, resolved hyperparameters plus selected Optuna parameters, final objective metrics, and artifact paths with hashes; result NPZ files store prediction scores together with sample IDs and fold metadata, while larger logs, rules, and plots remain in their normal files and are referenced from the manifest.

Optuna artifacts are organized by stage, outer fold, and tuning target:

```text
logs/optuna/<study-name>/<model>/nested/outer_<k>/<target>/study.db
logs/optuna/<study-name>/<model>/nested/outer_<k>/<target>/best_trial.toml
logs/optuna/<study-name>/<model>/nested/selected_params.toml
```

Nested evaluation output is organized under:

```text
logs/training/<dataset-names>/<group-names>/rrl/calibrated-fusion/
```

The important exported files include:

| File pattern | Meaning |
| --- | --- |
| `manifest.toml` | Compact provenance manifest for reproducing and auditing the run |
| `train.log` | Training log |
| `rrl_<name>_<outer>_<inner>.tsv` | Exported RRL rule table for one modality/fold |
| `rrl_<name>_<outer>_<inner>.preprocessing.toml` | Fold-fitted RRL parser preprocessing sidecar with retained availability columns and missingness-filter metadata |
| `rrl_<name>_<outer>_<inner>.feature-selection.toml` | Fold-fitted supervised feature-selection artifact when feature selection is enabled |
| `rrl_<name>_<outer>_<inner>.simple-imputer.toml` | Fold-fitted simple-imputer artifact when original missing-aware RRL mode is enabled |
| `train_avg_<result>.log` | Mean/std fold metrics |
| `train_ci_<result>.log` | Bootstrap confidence intervals |
| `results_<result>.npz` | Saved `y_true`, `y_pred`, `y_score`, `y_mask`, and fold metrics |
| `roc_<result>.png` | ROC curve with AUROC |

For `calibrated-fusion`, aggregate result names are:

```text
all_calibrated_fusion_original
all_calibrated_fusion_clinical_recall
all_calibrated_fusion_youden
```

For `calibrated-concat`, aggregate result names are:

```text
all_calibrated_concat_original
all_calibrated_concat_clinical_recall
all_calibrated_concat_youden
```

When `--no-clinical-threshold` is set, the `*_clinical_recall` result is omitted.

## Parser Re-evaluation

Downstream tools use the RRL parser rather than the live PyTorch model. Parser-based internal re-evaluation is an optional parser parity check against the live-model nested CV output; it reuses the live nested fold-level artifacts from `logs/training/<dataset-names>/<group-names>/rrl/calibrated-fusion/fold_artifacts/outer_<k>/available_fusion.toml` and does not publish artifacts for final external validation:

```bash
uv run iatreion train rrl-parser \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --no-pp \
  --log-root logs
```

Use `--no-pp` when the exported rules should be evaluated directly on processed feature values. In the current missing-aware RRL workflow, this is the usual choice for parser re-evaluation and external validation. Supervised feature selection is not re-fitted during parser evaluation; final rules access the selected features by column name.

Parser re-evaluation does not train a live RRL model and does not use a validation split, so `--val-size` belongs in live RRL training settings rather than `[train.rrl-parser]`.

This command writes parser-based parity metrics under:

```text
logs/training/<dataset-names>/<group-names>/rrl-parser/calibrated-fusion/
```

## Final Model Fitting

For external validation, run final tuning and final fitting on all internal data with `--tune-config -f`. The final command should match the internal nested-evaluation settings for source modalities, labels, groups, preprocessing, and aggregation; modality-subset fusion artifacts can then be published with result replay.

```bash
uv run iatreion train rrl \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
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
  --tune-config configs/optuna_rrl.toml \
  -f
```

Final tuning artifacts use the same fixed `{log-root}/optuna` root as nested tuning; final model and rule artifacts use `--log-root` or `train.rrl.log-root`.

Final tuning artifacts are written under:

```text
logs/optuna/<study-name>/<model>/final/<target>/study.db
logs/optuna/<study-name>/<model>/final/<target>/best_trial.toml
logs/optuna/<study-name>/<model>/final/selected_params.toml
```

The final runner fits an internal OOF `available_fusion.toml` with the selected final parameters under `logs/final-calibration/<dataset-names>/<group-names>/rrl/average/<dataset-names>/`, then publishes the full-modality artifact to:

```text
logs/final/<group-names>/rrl/fusion/<dataset-names>/available_fusion.toml
```

Final RRL artifacts are saved per modality as:

```text
logs/final/<group-names>/rrl/artifacts/<name>/rules.tsv
logs/final/<group-names>/rrl/artifacts/<name>/preprocessing.toml
logs/final/<group-names>/rrl/artifacts/<name>/feature-selection.toml
logs/final/<group-names>/rrl/artifacts/<name>/simple-imputer.toml
```

The `preprocessing.toml` sidecar is always written for RRL and tells the parser which retained raw features define modality availability; `feature-selection.toml` and `simple-imputer.toml` are written only when those preprocessing steps are enabled. Original RRL uses `simple-imputer.toml` for parser-time imputation, while improved missing-aware RRL keeps missing values and uses `preprocessing.toml` for availability without requiring a simple-imputer artifact.

## Result Replay Subset Fusion

`uv run iatreion train result-replay` reads saved `results_<name>.npz` probability files from a source model (`rrl`, `xgboost`, or `random-forest`), aligns samples by the embedded sample IDs, and refits calibrated-fusion artifacts for the requested modality subset without retraining the source models. `-n/--names` should list the source modalities whose result files define the run, `--eval-names` selects the subset to fuse, and `-a/--aggregate` selects the source result directory for internal replay; final replay infers the source aggregate from the final-calibration rules.

```bash
uv run iatreion train result-replay \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --eval-names symptom csvd \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --source-model rrl \
  --log-root logs
```

For internal evaluation, result replay fits each outer fold's subset fusion artifact from the corresponding inner-CV result NPZ files and evaluates that artifact on the outer fold; it also fits a global OOF artifact for later inspection. Output is written under `logs/training/<dataset-names>/<group-names>/result-replay/<source-model>/<aggregate>/<subset-key>/`, with the global artifact at `available_fusion.toml` and fold artifacts under `fold_artifacts/outer_<k>/available_fusion.toml`.

For final subset artifacts, first run final model fitting so `logs/final-calibration/<dataset-names>/<group-names>/<source-model>/average/<dataset-names>/results_<name>.npz` exists, then replay with `-f`:

```bash
uv run iatreion train result-replay \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --eval-names symptom csvd \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  --source-model rrl \
  --log-root logs \
  -f
```

Final result replay writes metrics and its run-local artifact under `logs/final-calibration/<dataset-names>/<group-names>/result-replay/<source-model>/<source-aggregate>/<subset-key>/`, then publishes the subset artifact to `logs/final/<group-names>/<source-model>/fusion/<subset-key>/available_fusion.toml`. Internal result replay metrics include non-stratified bootstrap confidence intervals in `train_ci_<result>.log`, and CI lower/upper bounds are stored in the run manifest; final result replay keeps point-estimate logs only because its main role is artifact publication. Samples missing all selected modalities are excluded from metric calculation.

## External Validation

Use `uv run iatreion eval rrl` to apply final RRL rule files to external data. The command uses:

- `--log-root`: root directory containing final RRL logs.
- `-p/--process`: internal `process_info.toml`, needed to reproduce feature encodings and category orders.
- `--data.<raw-data-name>`: external Excel, CSV, or TSV spreadsheet for each requested raw data source.
- `--data-sheets.<raw-data-name>`: optional Excel sheet name or index for a raw data source.
- `--index-name`: external sample ID column, required for modes that read external data.
- `--label-name`: external label column, required for `-m eval` and `-m rule-or`; in other modes it is optional and only used to exclude a label column from features.
- `--bootstrap-samples` and `--ci-level`: non-stratified bootstrap settings for labeled external-evaluation confidence intervals; defaults are 1000 resamples and 0.95.
- `-o/--output`: exported spreadsheet path for `batch`, `ranked-rules`, and `rule-or` modes; supported suffixes are `.xlsx`, `.csv`, and `.tsv`.

For labeled external evaluation, `eval.log` reports point estimates with bootstrap confidence intervals, and logs, ROC plots, and default ranked-rule TSV files are written under `logs/final/<group-names>/rrl/eval/<dataset-names>/`, so different modality lists do not overwrite each other.

Example for symptom plus CSVD:

```bash
uv run iatreion eval rrl \
  -n symptom csvd \
  --index-name "<sample-id-column>" \
  --label-name "<label-column>" \
  -g a c \
  --positive-label c \
  -t logs \
  -p "<path-to-the-process-info-file>" \
  --data.history "<path-to-the-spreadsheet-for-symptom-data>" \
  --data.csvd "<path-to-the-spreadsheet-for-csvd-data>" \
  -m eval \
  -k last \
  -D
```

Example including MRI volume features:

```bash
uv run iatreion eval rrl \
  -n symptom csvd volume-new-pct \
  --index-name "<sample-id-column>" \
  --label-name "<label-column>" \
  -g a c \
  --positive-label c \
  -t logs \
  -p "<path-to-the-process-info-file>" \
  --data.history "<path-to-the-spreadsheet-for-symptom-data>" \
  --data.csvd "<path-to-the-spreadsheet-for-csvd-data>" \
  --data.volume-new "<path-to-the-spreadsheet-for-mri-volume-data>" \
  -v "<path-to-the-excel-file-storing-mean-and-std-for-mri-volume-data>" \
  --vmri-change "<path-to-the-file-storing-column-name-changes-for-mri-volume-data>" \
  -m eval \
  -k last \
  -D
```

Evaluation modes:

| Mode | Meaning |
| --- | --- |
| `single` | Show prediction and active supporting/opposing rules for one sample |
| `batch` | Export predictions for a batch without metrics; set `-o/--output` to an `.xlsx`, `.csv`, or `.tsv` path |
| `eval` | Compute metrics when external labels are available |
| `rule-or` | Export per-rule unadjusted odds ratios against each rule's predicted label; set `-o/--output` to an `.xlsx`, `.csv`, or `.tsv` path |
| `show` | List exported model rules |
| `ranked-rules` | Export all non-bias rules across modalities, sorted by calibrated fusion score; set `-o/--output` to an `.xlsx`, `.csv`, or `.tsv` path |

For single-sample explanations, pass `--sample-id`.

`show` and `ranked-rules` report transformed rule scores that include the raw rule weights, each modality's calibration slope, and the modality fusion weight; the calibration intercept only affects the Bias term. When `-o/--output` is omitted, `batch` writes `rrl_batch_result.xlsx`, `ranked-rules` writes `rrl_ranked_rules.tsv` under `logs/final/<group-names>/rrl/eval/<dataset-names>/`, and `rule-or` writes `rrl_rule_or.tsv`.

## XGBoost and Random Forest Baselines

XGBoost and Random Forest can be trained with the same processed `.data` and `.info` files as RRL. Their defaults can live in `[train.xgboost]` and `[train.random-forest]` in `configs/config.toml`; common fields such as `prefix`, `names`, `label-name`, `groups`, `positive-label`, `aggregate`, `use-clinical-threshold`, `clinical-threshold-label`, `clinical-threshold-recall`, `log-root`, and `tune-config` follow the same config/CLI override rules described above. For external validation, train final baseline models with `-f -a calibrated-fusion` so the final log directory contains modality model artifacts, transform artifacts, and `fusion/<dataset-names>/available_fusion.toml`; use final result replay for any additional subset artifacts.

Baseline tuning uses the same nested/final Optuna workflow as RRL. `configs/optuna_xgboost.toml` tunes `num-round`, `learning-rate`, `max-depth`, `min-child-weight`, row/column subsampling, split loss, and L1/L2 regularization; `configs/optuna_random_forest.toml` tunes tree count, depth, leaf/split sizes, feature sampling, class weighting, bootstrap sample fraction, and cost-complexity pruning.

With the config file, run:

```bash
uv run iatreion train xgboost --config configs/config.toml
uv run iatreion train random-forest --config configs/config.toml
```

CLI arguments override the TOML values, so you can keep the common baseline setup in the file and change only the machine- or run-specific values:

```bash
uv run iatreion train xgboost --config configs/config.toml --tune-config configs/optuna_xgboost.toml -i 6-7 --device cuda
uv run iatreion train random-forest --config configs/config.toml -p "<path-to-the-folder-storing-processed-data>"
```

The equivalent direct commands are:

```bash
uv run iatreion train xgboost \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom csvd \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --log-root logs \
  -i 0-7 \
  --device cuda \
  --tune-config configs/optuna_xgboost.toml

uv run iatreion train random-forest \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom csvd \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
  --clinical-threshold-label c --clinical-threshold-recall 0.6 \
  -a calibrated-fusion \
  --log-root logs \
  --tune-config configs/optuna_random_forest.toml
```

Baseline internal outputs follow the same `{log-root}/training/<dataset-names>/<group-names>/<model>/<aggregate>/` directory convention as other training commands. Final artifacts used by `iatreion eval` are written under `logs/final/<group-names>/xgboost/` or `logs/final/<group-names>/random-forest/`, with per-modality artifacts in `artifacts/<name>/`; each `transform.toml` records the fitted preprocessing schema and embeds missingness-filter, feature-selection, or simple-imputer metadata when those steps are enabled, and external-validation `eval.log` files report point estimates with bootstrap confidence intervals under each final model root's `eval/<dataset-names>/` subdirectory.

### Baseline External Validation

Use `uv run iatreion eval xgboost` or `uv run iatreion eval random-forest` to apply final baseline models to external data. Baseline external validation always uses final calibrated-fusion artifacts: each requested modality must have `logs/final/<group-names>/<model>/artifacts/<name>/transform.toml` plus a saved model file, and `logs/final/<group-names>/<model>/fusion/<dataset-names>/available_fusion.toml` must exist for exactly the requested modality list. Labeled baseline external validation uses the same non-stratified bootstrap CI controls as RRL eval: `--bootstrap-samples` and `--ci-level`. The transform artifact contains the fitted preprocessing schema and, when enabled, embedded missingness-filter, feature-selection, and simple-imputer metadata. If the subset artifact is missing, the command fails instead of falling back to uncalibrated averaging. Final baseline fitting with `-f -a calibrated-fusion` writes the full-modality artifact; use final result replay to publish additional modality subsets.

Example for labeled external XGBoost validation:

```bash
uv run iatreion eval xgboost \
  -n symptom csvd volume-new-pct \
  --index-name "<sample-id-column>" \
  --label-name "<label-column>" \
  -g a c \
  --positive-label c \
  --log-root logs \
  -p "<path-to-the-process-info-file>" \
  --data.history "<path-to-the-spreadsheet-for-symptom-data>" \
  --data.csvd "<path-to-the-spreadsheet-for-csvd-data>" \
  --data.volume-new "<path-to-the-spreadsheet-for-mri-volume-data>" \
  -v "<path-to-the-excel-file-storing-mean-and-std-for-mri-volume-data>" \
  --vmri-change "<path-to-the-file-storing-column-name-changes-for-mri-volume-data>" \
  -m eval \
  -k last
```

For unlabeled batch prediction, use `-m batch`; when `-o/--output` is omitted, the command writes `baseline_batch_result.xlsx`.

## Figures and Tables

Use `uv run iatreion show` for manuscript tables and figures.

Common outputs:

```bash
uv run iatreion show table-1 \
  -p "<path-to-the-folder-storing-processed-data>" \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  --label-name group_encrypted \
  -g a c \
  --positive-label c \
  -o table_1
```

```bash
uv run iatreion show latex-ci-delong \
  -n symptom s-screen-sum composite-bin biomarker cbf csvd volume-new-pct \
  -g a c \
  --positive-label c \
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
  --positive-label c \
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

## GUI

After final model fitting, launch:

```bash
uv run iatreion-gui
```

The GUI wraps the same parser API used by `iatreion eval rrl`. It can load final RRL models, select input files, export batch predictions and rule-OR tables, view active rules, and evaluate labeled external data.

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

Confidence intervals use non-stratified bootstrap resampling of the evaluated samples. Training, result replay, and labeled external validation all use the same defaults:

```text
--bootstrap-samples 1000
--ci-level 0.95
```

Plot/table helpers include:

- Wilcoxon signed-rank tests over fold metrics.
- DeLong tests over full out-of-fold AUROCs.
- McNemar tests for paired accuracy comparisons.

For binary labels, set `--positive-label` in each analysis. Internally, the configured positive label is encoded as class index `1`; AUROC and AUPRC use that positive-class score, and AUPRC is reported as sklearn average precision rather than interpolated trapezoidal PR-AUC.

## Repository Layout

```text
configs/
  config.toml                      # Main CLI defaults
  optuna_rrl.toml                  # RRL Optuna search-space defaults
  optuna_xgboost.toml              # XGBoost Optuna search-space defaults
  optuna_random_forest.toml        # Random Forest Optuna search-space defaults
scripts/
  pipeline.sh                      # Manuscript modeling/evaluation command sequence
src/iatreion/
  cli/                             # CLI commands
  configs/                         # CLI dataclasses and config semantics
  preprocessors/                   # Raw-to-processed feature extraction
  train_utils/                     # Splitting, feature selection, encoding, fusion, imputation artifacts
  trainers/                        # Training loop, metric recording, provenance manifests, final artifacts
  models/rrl.py                    # Live RRL training wrapper
  models/rrl_discrete.py           # Parser/evaluator for exported RRL rules
  rrl/                             # RRL network and rule export implementation
  show_helpers/                    # Manuscript figure/table helpers
  gui/                             # Tk GUI for final RRL evaluation
```

## Troubleshooting

### Available-fusion artifact not found

Run final training before external validation. For RRL, use `uv run iatreion train rrl ... -f`; for baselines, use `uv run iatreion train xgboost ... -f` or `uv run iatreion train random-forest ... -f`. The external resolver directly looks for `logs/final/<group-names>/<model>/fusion/<dataset-names>/available_fusion.toml`; run `uv run iatreion train result-replay ... --eval-names <subset> -f` to publish subset artifacts.

### No experiment root found

Check that `-p`, `-n`, `-g`, `--positive-label`, `-a`, and `--log-root` match the previous training command, and that the requested `-n` list has a matching final `fusion/<dataset-names>/available_fusion.toml`.

### External validation cannot find a column

Confirm the external index column with `--index-name`, label column with `--label-name`, and raw data mapping such as `--data.history ...`, `--data.csvd ...`, or `--data.volume-new ...`. MRI volume validation also needs the fixed Excel mean/std workbook via `-v` and a column-change spreadsheet via `--vmri-change`.

### Parser metrics use fewer samples than expected

Samples with all selected modalities missing are masked out. If using `--eval-names symptom csvd`, any sample missing both symptom and CSVD is excluded from metric calculation.
