from rich import box
from rich.table import Column, Table

from iatreion.api import get_batched_result, get_eval_result, get_models, get_result
from iatreion.configs import RrlEvalConfig
from iatreion.utils import logger

from .common import console


def get_table(title: str, *headers: str) -> Table:
    right_columns = {'Score', 'Probability', 'Confidence', 'Weight'}
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
    result_list, pred_list, bias_list, support_list, oppose_list = get_result(config)

    result_table = get_table('Result', 'Label', 'Probability', 'Confidence')
    result_table.add_row(*result_list[0], style='bold green')
    console.print(result_table)

    pred_table = get_table(
        'Predictions', 'Module', 'Label', 'Probability', 'Confidence', 'Weight'
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
            *row[1:-2],
            f'{row.Probability:.2%}',
            f'{row.Confidence:.2%}',
        )
    console.print(result_table)


def display_eval_result(config: RrlEvalConfig) -> None:
    result, fig, model_config = get_eval_result(config)
    model_config.register_log_dir('rrl-eval', file_name='eval.log')
    logger.info(result)
    if fig is not None:
        fig.savefig(model_config.train.roc_file, dpi=300)


def display_models(config: RrlEvalConfig) -> None:
    rule_list = get_models(config)
    for name, rules in rule_list:
        table = get_table(name, 'Label', 'Score', 'Rule')
        table.add_row(*rules[0], 'Initial Bias', style='yellow')
        for line in rules[1:]:
            table.add_row(*line)
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
