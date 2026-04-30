from collections import defaultdict

from rich import box
from rich.table import Column, Table

from iatreion.api import (
    RrlTermOption,
    get_batched_result,
    get_eval_result,
    get_result,
    get_rule_options,
)
from iatreion.configs import RrlEvalConfig
from iatreion.utils import logger

from .common import console


def get_table(title: str, *headers: str) -> Table:
    right_columns = {
        'Boundary',
        'Positive Probability',
        'Probability',
        'Score',
        'Threshold',
        'Weight',
    }
    return Table(
        *(
            Column(header=name, justify='right' if name in right_columns else 'left')
            for name in headers
        ),
        title=title,
        box=box.ROUNDED,
        title_style='italic yellow',
    )


def display_result(config: RrlEvalConfig) -> None:
    (
        sample_id,
        result_list,
        pred_list,
        bias_list,
        support_list,
        oppose_list,
    ) = get_result(config)

    result_table = get_table(
        f'Result (sample={sample_id})',
        'Label',
        'Score',
        'Boundary',
        'Probability',
        'Positive Probability',
        'Threshold',
    )
    result_table.add_row(*result_list[0], style='bold green')
    console.print(result_table)

    pred_table = get_table(
        'Predictions',
        'Module',
        'Label',
        'Score',
        'Probability',
        'Weight',
    )
    for line in pred_list:
        pred_table.add_row(*line)
    console.print(pred_table)

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
        result_table.add_row(
            str(row.Index),
            *[str(value) for value in row[1:-1]],
            f'{row.Probability:.2%}',
        )
    console.print(result_table)


def display_eval_result(config: RrlEvalConfig) -> None:
    result, fig, model_config = get_eval_result(config)
    model_config.register_log_dir('rrl-eval', file_name='eval.log')
    logger.info(result)
    if fig is not None:
        dataset, train = model_config.dataset, model_config.train
        fig.savefig(train.get_roc_file(dataset.name_str), dpi=300)


def display_models(config: RrlEvalConfig) -> None:
    options_by_name: dict[str, list[RrlTermOption]] = defaultdict(list)
    for option in get_rule_options(config):
        options_by_name[option.module].append(option)
    for name, options in options_by_name.items():
        table = get_table(name, 'Index', 'Label', 'Score', 'Rule')
        for option in options:
            table.add_row(
                option.display_index,
                option.label,
                f'{option.score:.2f}',
                option.rule,
                style='yellow' if option.kind == 'bias' else None,
            )
        console.print(table)


def rrl_eval(*, config: RrlEvalConfig | None = None) -> None:
    """Evaluate an RRL model."""
    if config is None:
        config = RrlEvalConfig()
    match config.mode:
        case 'single':
            display_result(config)
        case 'batch':
            display_batched_result(config)
        case 'eval':
            display_eval_result(config)
        case 'show':
            display_models(config)
