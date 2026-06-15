#!/usr/bin/env bash

set -euxo pipefail

config_path="configs/config.toml"
run_process=true

prefix="<path-to-the-folder-storing-processed-data>"
process_harmonized="<path-to-the-internal-harmonized-spreadsheet>"
eval_harmonized="<path-to-the-external-harmonized-spreadsheet>"
swap_harmonized=false
log_root_suffix=""

if [[ "$swap_harmonized" == true ]]; then
    harmonized_tmp="$process_harmonized"
    process_harmonized="$eval_harmonized"
    eval_harmonized="$harmonized_tmp"
    log_root_suffix="_swapped"
fi

process_info="${prefix}/process_info.toml"
imputed_log_root="logs_imputed${log_root_suffix}"
not_imputed_log_root="logs_not_imputed${log_root_suffix}"

process_path_args=(
    --prefix "$prefix"
    --data.harmonized "$process_harmonized"
)

train_path_args=(
    --prefix "$prefix"
)

eval_path_args=(
    --process "$process_info"
    --data.harmonized "$eval_harmonized"
)

eval_subsets=(
    "h-demo h-mmse h-moca h-mri h-history sh-apoe-labdata"
    "h-demo h-mmse h-moca h-mri-roi h-history sh-apoe-labdata"
)

baseline_models=(xgboost random-forest)
rrl_models=(rrl)

tasks=(
    "A|A+ A-|A+"
    "T|T+ T-|T+"
    "MMSE_progression_group|fast slow|fast"
)

iatreion() {
    uv run iatreion --config "$config_path" "$@"
}

process() {
    iatreion process "${process_path_args[@]}"
}

build_task_args() {
    local -n args_ref="${1}"
    local label_name="${2}"
    local groups="${3}"
    local positive_label="${4}"
    local log_root="${5}"
    local -a group_values=()

    read -r -a group_values <<< "$groups"
    args_ref=(
        --label-name "$label_name"
        --groups "${group_values[@]}"
        --positive-label "$positive_label"
        --log-root "$log_root"
    )
}

train_eval() {
    local -n models_ref="${1}"
    local -n task_args_ref="${2}"
    local -n train_args_ref="${3}"
    local model_name subset
    local -a subset_names=()

    for model_name in "${models_ref[@]}"; do
        iatreion train "$model_name" "${train_path_args[@]}" "${task_args_ref[@]}" "${train_args_ref[@]}"
        iatreion train "$model_name" "${train_path_args[@]}" "${task_args_ref[@]}" "${train_args_ref[@]}" -f

        iatreion train result-replay "${train_path_args[@]}" "${task_args_ref[@]}" --source-model "$model_name"

        for subset in "${eval_subsets[@]}"; do
            read -r -a subset_names <<< "$subset"
            iatreion train result-replay "${train_path_args[@]}" "${task_args_ref[@]}" --source-model "$model_name" \
                --eval-names "${subset_names[@]}"
            iatreion train result-replay "${train_path_args[@]}" "${task_args_ref[@]}" --source-model "$model_name" \
                --eval-names "${subset_names[@]}" -f
        done

        for subset in "${eval_subsets[@]}"; do
            read -r -a subset_names <<< "$subset"
            iatreion eval "$model_name" "${eval_path_args[@]}" "${task_args_ref[@]}" -n "${subset_names[@]}"
        done
    done
}

parity_check() {
    local -n task_args_ref="${1}"
    iatreion train rrl-parser "${train_path_args[@]}" "${task_args_ref[@]}"
}

run_baselines_for_task() {
    local label_name="${1}"
    local groups="${2}"
    local positive_label="${3}"
    local -a task_args=()
    local -a train_args=()

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$imputed_log_root"
    train_args=(--importance-methods native)
    train_eval baseline_models task_args train_args

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$not_imputed_log_root"
    train_args=(--importance-methods native --missing-value-strategy none)
    train_eval baseline_models task_args train_args
}

run_rrl_for_task() {
    local label_name="${1}"
    local groups="${2}"
    local positive_label="${3}"
    local -a task_args=()
    local -a train_args=()

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$imputed_log_root"
    train_args=()
    train_eval rrl_models task_args train_args

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$not_imputed_log_root"
    train_args=(--missing-aware-mode improved)
    train_eval rrl_models task_args train_args

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$imputed_log_root"
    parity_check task_args

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$not_imputed_log_root"
    parity_check task_args
}

task() {
    local label_name="${1}"
    local groups="${2}"
    local positive_label="${3}"

    run_baselines_for_task "$label_name" "$groups" "$positive_label"
    run_rrl_for_task "$label_name" "$groups" "$positive_label"
}

run_pipeline() {
    local task_spec label_name groups positive_label

    if [[ "$run_process" == true ]]; then
        process
    fi

    for task_spec in "${tasks[@]}"; do
        IFS='|' read -r label_name groups positive_label <<< "$task_spec"
        task "$label_name" "$groups" "$positive_label"
    done
}

run_pipeline
