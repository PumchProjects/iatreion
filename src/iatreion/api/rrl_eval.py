import numpy as np
import pandas as pd

from iatreion.configs import RrlEvalConfig
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessor


def get_max_label(arr: list[float], labels: list[str]) -> str:
    return labels[np.argmax(arr).item()]


def calc_score(arr: list[float]) -> float:
    return max(arr) - min(arr)


def get_data_model(config: RrlEvalConfig) -> tuple[pd.DataFrame, DiscreteRrlModel]:
    process_config, rrl_config = config.make_configs()
    preprocessor = get_preprocessor(process_config)
    data = preprocessor.get_data()
    model = DiscreteRrlModel(rrl_config)
    return data, model


def get_result(config: RrlEvalConfig) -> tuple[list[list[str]], ...]:
    data, model = get_data_model(config)
    result, active_lines, rrl = model.interpret(data)
    max_label = get_max_label(result, rrl.labels)
    result_list = [[max_label, f'{calc_score(result):.2f}']]
    bias_max_label = get_max_label(rrl.biases, rrl.labels)
    bias_list = [[bias_max_label, f'{calc_score(rrl.biases):.2f}']]
    support_list: list[list[str]] = []
    oppose_list: list[list[str]] = []
    for line in active_lines:
        weight_max_label = get_max_label(line.weights, rrl.labels)
        score = calc_score(line.weights)
        if weight_max_label == max_label:
            support_list.append([weight_max_label, f'{score:.2f}', line.print_rule()])
        else:
            oppose_list.append([weight_max_label, f'{score:.2f}', line.print_rule()])
    return result_list, bias_list, support_list, oppose_list


def get_batched_result(config: RrlEvalConfig) -> pd.DataFrame:
    data, model = get_data_model(config)
    result, rrl = model.eval(data)
    y_pred = [rrl.labels[i] for i in result.argmax(axis=1)]
    y_score = result.max(axis=1) - result.min(axis=1)
    df = pd.DataFrame({'Label': y_pred, 'Score': y_score}, index=data.index)
    return df
