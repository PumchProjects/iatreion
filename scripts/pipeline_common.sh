#!/usr/bin/env bash

process_prefix="${prefix}/processed"
process_info="${process_prefix}/process_info.toml"

if [[ "$swap_harmonized" == true ]]; then
    harmonized_tmp="$process_harmonized"
    process_harmonized="$eval_harmonized"
    eval_harmonized="$harmonized_tmp"
    log_root_suffix="_swapped"
fi

imputed_log_root="${prefix}/logs_imputed${log_root_suffix}"
not_imputed_log_root="${prefix}/logs_not_imputed${log_root_suffix}"

process_path_args=(
    --prefix "$process_prefix"
    --data.harmonized "$process_harmonized"
)

train_path_args=(
    --prefix "$process_prefix"
)

eval_path_args=(
    --process "$process_info"
    --data.harmonized "$eval_harmonized"
)

eval_subsets=(
    "h-demo h-mmse h-moca h-mri h-history sh-apoe-labdata"
    "h-demo h-mmse h-moca h-mri-roi h-history sh-apoe-labdata"
)

tasks=(
    "A|A+ A-|A+"
    "T|T+ T-|T+"
    "MMSE_progression_group|fast slow|fast"
)

iatreion() {
    uv run --no-sync iatreion --config "$config_path" "$@"
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
    local -a eval_model_args=()
    local -a subset_names=()

    for model_name in "${models_ref[@]}"; do
        eval_model_args=()
        if [[ "$model_name" == tabpfn ]]; then
            eval_model_args=(--model-path "$tabpfn_model_path")
        fi

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
            iatreion eval "$model_name" "${eval_path_args[@]}" "${task_args_ref[@]}" \
                "${eval_model_args[@]}" -n "${subset_names[@]}" -m eval
        done
    done
}

eval_rrl_ranked_rules() {
    local -n task_args_ref="${1}"
    local subset
    local -a subset_names=()

    for subset in "${eval_subsets[@]}"; do
        read -r -a subset_names <<< "$subset"
        iatreion eval rrl "${eval_path_args[@]}" "${task_args_ref[@]}" -n "${subset_names[@]}" -m ranked-rules
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
    if ((${#baseline_models[@]})); then
        train_args=(--importance-methods native)
        train_eval baseline_models task_args train_args
    fi
    if ((${#tabpfn_models[@]})); then
        train_args=(--model-path "$tabpfn_model_path")
        train_eval tabpfn_models task_args train_args
    fi

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$not_imputed_log_root"
    if ((${#nan_baseline_models[@]})); then
        train_args=(--importance-methods native --missing-value-strategy none)
        train_eval nan_baseline_models task_args train_args
    fi
    if ((${#tabpfn_models[@]})); then
        train_args=(--model-path "$tabpfn_model_path" --missing-value-strategy none)
        train_eval tabpfn_models task_args train_args
    fi
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
    eval_rrl_ranked_rules task_args
    parity_check task_args

    build_task_args task_args "$label_name" "$groups" "$positive_label" "$not_imputed_log_root"
    train_args=(--missing-aware-mode improved)
    train_eval rrl_models task_args train_args
    eval_rrl_ranked_rules task_args
    parity_check task_args
}

task() {
    local label_name="${1}"
    local groups="${2}"
    local positive_label="${3}"

    if ((${#baseline_models[@]} || ${#tabpfn_models[@]})); then
        run_baselines_for_task "$label_name" "$groups" "$positive_label"
    fi
    if ((${#rrl_models[@]})); then
        run_rrl_for_task "$label_name" "$groups" "$positive_label"
    fi
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
