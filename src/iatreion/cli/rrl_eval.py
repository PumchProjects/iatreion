from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rich import box
from rich.table import Table

from iatreion.configs import DiscreteRrlConfig, PreprocessorConfig
from iatreion.models import DiscreteRrlModel, Line, Rrl
from iatreion.preprocessors import get_preprocessor

from .common import app, console


def get_max_label(arr: list[float], labels: list[str]) -> str:
    return labels[np.argmax(arr).item()]


def calc_score(arr: list[float]) -> float:
    return max(arr) - min(arr)


def get_table(title: str, *headers: str) -> Table:
    return Table(
        *headers,
        title=title,
        box=box.ROUNDED,
        title_style='italic yellow',
    )


def display_results(result: list[float], active_lines: list[Line], rrl: Rrl) -> None:
    max_label = get_max_label(result, rrl.labels)
    result_table = get_table('Result', 'Label', 'Score')
    result_table.add_row(max_label, f'{calc_score(result):.2f}', style='bold green')
    console.print(result_table)
    bias_table = get_table('Initial Bias', 'Label', 'Score')
    bias_max_label = get_max_label(rrl.biases, rrl.labels)
    bias_table.add_row(bias_max_label, f'{calc_score(rrl.biases):.2f}')
    console.print(bias_table)
    support_table = get_table('Supporting Rules', 'Label', 'Score', 'Rule')
    oppose_table = get_table('Opposing Rules', 'Label', 'Score', 'Rule')
    for line in active_lines:
        weight_max_label = get_max_label(line.weights, rrl.labels)
        score = calc_score(line.weights)
        if weight_max_label == max_label:
            support_table.add_row(weight_max_label, f'{score:.2f}', line.print_rule())
        else:
            oppose_table.add_row(weight_max_label, f'{score:.2f}', line.print_rule())
    console.print(support_table)
    console.print(oppose_table)


def display_batched_results(results: NDArray, rrl: Rrl) -> None:
    y_pred = [rrl.labels[i] for i in results.argmax(axis=1)]
    y_score = (results.max(axis=1) - results.min(axis=1)).tolist()
    result_table = get_table('Result', 'Label', 'Score')
    for label, score in zip(y_pred, y_score, strict=False):
        result_table.add_row(label, f'{score:.2f}')
    console.print(result_table)


@app.command(sort_key=2)
def rrl_eval(*, config: DiscreteRrlConfig) -> None:
    """Evaluate an RRL model."""
    config.dataset.simple = True
    config.train.final = True
    process_config = PreprocessorConfig(dataset=config.dataset, output_prefix=Path())
    preprocessor = get_preprocessor(process_config)
    data = preprocessor.get_data()
    model = DiscreteRrlModel(config)
    if config.batched:
        results, rrl = model.eval(data)
        display_batched_results(results, rrl)
    else:
        result, active_lines, rrl = model.interpret(data)
        display_results(result, active_lines, rrl)
