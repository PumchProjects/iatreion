#!/usr/bin/env bash

set -euxo pipefail

export TABPFN_NO_BROWSER=1

config_path="configs/config.toml"
run_process=true

prefix="<path-to-the-folder-storing-results>"
process_harmonized="<path-to-the-internal-harmonized-spreadsheet>"
eval_harmonized="<path-to-the-external-harmonized-spreadsheet>"
tabpfn_model_path="<path-to-the-tabpfn-v3-classifier-checkpoint>"
swap_harmonized=false
log_root_suffix=""

baseline_models=(xgboost random-forest logistic-regression c45 cart)
nan_baseline_models=(xgboost random-forest)
tabpfn_models=(tabpfn)
rrl_models=(rrl)

source scripts/pipeline_common.sh
