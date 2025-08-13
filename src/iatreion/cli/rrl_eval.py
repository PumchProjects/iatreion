from pathlib import Path

import pandas as pd
from rich.panel import Panel

from iatreion.configs import DiscreteRrlConfig, PreprocessorConfig
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessor

from .common import app


def display_results(result: pd.DataFrame, active_rules: list[list[str]]) -> None:
    max_index = result.values.argmax().item()
    result_list = [f'{label}\t{result[label].item():.2%}' for label in result.columns]
    result_list[max_index] = f'[bold green]{result_list[max_index]}'
    result_str = '\n'.join(result_list)
    assert app.console is not None
    app.console.print(Panel(result_str, title='Results', title_align='left'))
    supporting_rules = [
        f'{result.columns[max_index]}\t{rule}' for rule in active_rules[max_index]
    ]
    app.console.print(
        Panel('\n'.join(supporting_rules), title='Supporting Rules', title_align='left')
    )
    opposing_rules: list[str] = []
    for i, rules in enumerate(active_rules):
        if i != max_index:
            opposing_rules += [f'{result.columns[i]}\t{rule}' for rule in rules]
    app.console.print(
        Panel('\n'.join(opposing_rules), title='Opposing Rules', title_align='left')
    )


@app.command(sort_key=2)
def rrl_eval(*, config: DiscreteRrlConfig) -> None:
    """Evaluate an RRL model."""
    config.dataset.simple = True
    config.train.final = True
    process_config = PreprocessorConfig(dataset=config.dataset, output_prefix=Path())
    preprocessor = get_preprocessor(process_config)
    data = preprocessor.get_data()
    model = DiscreteRrlModel(config)
    result, active_rules = model.interpret(data)
    display_results(result, active_rules)
