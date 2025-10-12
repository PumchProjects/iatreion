from typing import overload

import numpy as np
import pandas as pd

from iatreion.configs import RrlEvalConfig
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessor


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


def get_data_model(config: RrlEvalConfig) -> tuple[pd.DataFrame, DiscreteRrlModel]:
    process_config, rrl_config = config.make_configs()
    preprocessor = get_preprocessor(process_config)
    data = preprocessor.get_data_outer(add_indices=True)
    model = DiscreteRrlModel(rrl_config)
    return data, model


def get_result(config: RrlEvalConfig) -> tuple[list[list[str]], ...]:
    data, model = get_data_model(config)
    result, active_lines, rrl = model.interpret(data)
    max_label = get_max_label(result).item()
    result_list = [[max_label, f'{calc_score(result).item():.2f}']]
    bias_max_label = get_max_label(rrl.biases, rrl.labels)
    bias_list = [[bias_max_label, f'{calc_score(rrl.biases):.2f}']]
    support_list: list[list[str]] = []
    oppose_list: list[list[str]] = []
    if max_label:
        for line in active_lines:
            weight_max_label = get_max_label(line.weights, rrl.labels)
            score = calc_score(line.weights)
            if weight_max_label == max_label:
                support_list.append(
                    [weight_max_label, f'{score:.2f}', line.print_rule()]
                )
            else:
                oppose_list.append(
                    [weight_max_label, f'{score:.2f}', line.print_rule()]
                )
    return result_list, bias_list, support_list, oppose_list


def get_batched_result(config: RrlEvalConfig) -> pd.DataFrame:
    data, model = get_data_model(config)
    result = model.eval(data)
    y_pred = get_max_label(result)
    y_score = calc_score(result)
    df = pd.DataFrame({'Label': y_pred, 'Score': y_score})
    return df
