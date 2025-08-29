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
    result_list, bias_list, support_list, oppose_list = get_result(config)

    result_table = get_table('Result', 'Label', 'Score')
    result_table.add_row(*result_list[0], style='bold green')
    console.print(result_table)

    bias_table = get_table('Initial Bias', 'Label', 'Score')
    bias_table.add_row(*bias_list[0])
    console.print(bias_table)

    support_table = get_table('Supporting Rules', 'Label', 'Score', 'Rule')
    for line in support_list:
        support_table.add_row(*line)
    console.print(support_table)

    oppose_table = get_table('Opposing Rules', 'Label', 'Score', 'Rule')
    for line in oppose_list:
        oppose_table.add_row(*line)
    console.print(oppose_table)


def display_batched_result(config: RrlEvalConfig) -> None:
    result = get_batched_result(config)
    result_table = get_table('Result', 'ID', 'Label', 'Score')
    for index, label, score in zip(
        result.index, result['Label'], result['Score'], strict=False
    ):
        result_table.add_row(str(index), label, f'{score:.2f}')
    console.print(result_table)


@app.command(sort_key=2)
def rrl_eval(*, config: RrlEvalConfig) -> None:
    """Evaluate an RRL model."""
    if config.batched:
        display_batched_result(config)
    else:
        display_result(config)
