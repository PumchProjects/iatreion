from typing import overload

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from iatreion.configs import DiscreteRrlConfig, RrlEvalConfig
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessors
from iatreion.rrl import make_data_labels
from iatreion.trainers import Recorder


@overload
def get_max_label(arr: list[float], labels: list[str]) -> str: ...


@overload
def get_max_label(arr: pd.DataFrame) -> 'pd.Series[str]': ...


def get_max_label(
    arr: list[float] | pd.DataFrame, labels: list[str] | None = None
) -> 'str | pd.Series[str]':
    if isinstance(arr, list):
        assert labels is not None
        return labels[np.argmax(arr).item()]
    else:
        max_labels = arr.fillna(0).idxmax(axis=1, skipna=False).astype(str)
        max_labels.loc[arr.isna().all(axis=1)] = ''
        return max_labels


@overload
def calc_score(arr: list[float]) -> float: ...


@overload
def calc_score(arr: pd.DataFrame) -> 'pd.Series[float]': ...


def calc_score(arr: list[float] | pd.DataFrame) -> 'float | pd.Series[float]':
    if isinstance(arr, list):
        return max(arr) - min(arr)
    else:
        return (arr.max(axis=1) - arr.min(axis=1)).astype(float)


def get_models(config: RrlEvalConfig) -> list[tuple[str, list[str], list[list[str]]]]:
    _, rrl_config = config.make_configs()
    model = DiscreteRrlModel(rrl_config)
    names = rrl_config.dataset.names
    models = model.get_models()
    rule_list: list[tuple[str, list[str], list[list[str]]]] = []
    for name, rrl in zip(names, models, strict=False):
        rules: list[list[str]] = [[f'{bias:.2f}' for bias in rrl.biases]]
        for line in rrl.lines:
            weights = [f'{weight:.2f}' for weight in line.weights]
            rules.append([*weights, line.print_rule()])
        rule_list.append((name, rrl.labels, rules))
    return rule_list


def get_data_model(
    config: RrlEvalConfig,
) -> tuple[
    list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame | None, DiscreteRrlModel
]:
    process_config, rrl_config = config.make_configs()
    preprocessors = get_preprocessors(process_config)
    data = [preprocessor.get_data_outer() for preprocessor in preprocessors]
    additional_data = process_config.final_indices
    group_names = preprocessors[0].get_group_names() if process_config.eval else None
    model = DiscreteRrlModel(rrl_config)
    return data, additional_data, group_names, model


def get_result(config: RrlEvalConfig) -> tuple[list[list[str]], ...]:
    data, _, _, model = get_data_model(config)
    names, models, predictions, active_lines, result, confidence = model.interpret(data)
    max_label = get_max_label(result).item()
    result_list = [
        [max_label, f'{calc_score(result).item():.2f}', f'{confidence.item():.2%}']
    ]
    score_list: list[list[str]] = []
    for name, rrl, (pred, conf) in zip(names, models, predictions, strict=False):
        pred_max_label = get_max_label(pred).item()
        pred_score = calc_score(pred).item()
        score_list.append(
            [
                name,
                pred_max_label,
                f'{pred_score:.2f}',
                f'{conf.item():.2%}',
                f'{rrl.weight:.2%}',
            ]
        )
    bias_list: list[list[str]] = []
    for name, rrl in zip(names, models, strict=False):
        bias_max_label = get_max_label(rrl.biases, rrl.labels)
        bias_score = calc_score(rrl.biases)
        bias_list.append([name, bias_max_label, f'{bias_score:.2f}'])
    support_list: list[list[str]] = []
    oppose_list: list[list[str]] = []
    if max_label:
        for name, line in active_lines:
            weight_max_label = get_max_label(line.weights, line.labels)
            score = calc_score(line.weights)
            rule_list = [name, weight_max_label, f'{score:.2f}', line.print_rule()]
            if weight_max_label == max_label:
                support_list.append(rule_list)
            else:
                oppose_list.append(rule_list)
    return result_list, score_list, bias_list, support_list, oppose_list


def get_batched_result(config: RrlEvalConfig) -> pd.DataFrame:
    data, additional_data, _, model = get_data_model(config)
    result, confidence = model.eval(data)
    y_pred = get_max_label(result)
    y_pred.name = 'Label'
    y_score = calc_score(result)
    y_score.name = 'Score'
    confidence.name = 'Confidence'
    df = pd.concat(additional_data + [y_pred, y_score, confidence], axis=1)
    return df


def get_eval_result(
    config: RrlEvalConfig,
) -> tuple[str, Figure | None, DiscreteRrlConfig]:
    data, _, group_names, model = get_data_model(config)
    assert group_names is not None
    result, _ = model.eval(data)
    result = pd.concat([result, group_names], axis=1)
    train_config = model.config.train
    # Only select data in the target groups
    X_df, y_df = make_data_labels(result, train_config, group_names.columns.to_list())
    # Drop predictions that are failed
    y_df = y_df[~X_df.isna().all(axis=1)]
    X_df = X_df.dropna(how='all')
    y_true = y_df.map(train_config.get_group_index_mapping()).to_numpy()
    y_score = X_df.to_numpy('float32')
    index = X_df.index.to_numpy()
    recorder = Recorder(train_config)
    eval_result = recorder.record((0.0, y_true, y_score, index, {}))
    fig = recorder.roc.fig if train_config.plot_roc else None
    return eval_result, fig, model.config
