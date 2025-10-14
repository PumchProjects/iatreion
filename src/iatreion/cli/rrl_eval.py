from rich import box
from rich.table import Table

from iatreion.api import get_batched_result, get_result
from iatreion.configs import RrlEvalConfig

from .common import app, console


def get_table(title: str, *headers: str) -> Table:
    return Table(
        *headers,
        title=title,
        box=box.ROUNDED,
        title_style='italic yellow',
    )


def display_result(config: RrlEvalConfig) -> None:
    result_list, score_list, bias_list, support_list, oppose_list = get_result(config)

    result_table = get_table('Result', 'Label', 'Score')
    result_table.add_row(*result_list[0], style='bold green')
    console.print(result_table)

    score_table = get_table('Scores', 'Module', 'Label', 'Score')
    for line in score_list:
        score_table.add_row(*line)
    console.print(score_table)

    bias_table = get_table('Initial Biases', 'Module', 'Label', 'Score')
    for line in bias_list:
        bias_table.add_row(*line)
    console.print(bias_table)

    support_table = get_table('Supporting Rules', 'Module', 'Label', 'Score', 'Rule')
    for line in support_list:
        support_table.add_row(*line)
    console.print(support_table)

    oppose_table = get_table('Opposing Rules', 'Module', 'Label', 'Score', 'Rule')
    for line in oppose_list:
        oppose_table.add_row(*line)
    console.print(oppose_table)


def display_batched_result(config: RrlEvalConfig) -> None:
    result = get_batched_result(config)
    result_table = get_table('Result', 'ID', *result.columns)
    for row in result.itertuples():
        result_table.add_row(str(row.Index), *row[1:-1], f'{row.Score:.2f}')
    console.print(result_table)


@app.command(sort_key=2)
def rrl_eval(*, config: RrlEvalConfig) -> None:
    """Evaluate an RRL model."""
    if config.batched:
        display_batched_result(config)
    else:
        display_result(config)
