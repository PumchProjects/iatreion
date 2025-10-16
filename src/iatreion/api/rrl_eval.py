from typing import overload

import numpy as np
import pandas as pd

from iatreion.configs import RrlEvalConfig
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessors


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


def get_data_model(
    config: RrlEvalConfig,
) -> tuple[
    list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame | None, DiscreteRrlModel
]:
    process_config, rrl_config = config.make_configs()
    preprocessors = get_preprocessors(process_config)
    data = [preprocessor.get_data_outer() for preprocessor in preprocessors]
    additional_data = process_config.final_indices
    group_names = preprocessors[0].get_group_names() if config.debug else None
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


def get_batched_result(
    config: RrlEvalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, DiscreteRrlModel]:
    data, additional_data, group_names, model = get_data_model(config)
    result, confidence = model.eval(data)
    y_pred = get_max_label(result)
    y_pred.name = 'Label'
    y_score = calc_score(result)
    y_score.name = 'Score'
    confidence.name = 'Confidence'
    df = pd.concat(additional_data + [y_pred, y_score, confidence], axis=1)
    return df, result, group_names, model
