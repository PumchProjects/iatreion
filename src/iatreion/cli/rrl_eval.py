import pandas as pd
from rich import box
from rich.table import Table
from scipy.special import softmax

from iatreion.api import get_batched_result, get_result
from iatreion.configs import RrlEvalConfig
from iatreion.models import DiscreteRrlModel
from iatreion.rrl import make_data_labels
from iatreion.trainers import Recorder

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

    result_table = get_table('Result', 'Label', 'Score', 'Confidence')
    result_table.add_row(*result_list[0], style='bold green')
    console.print(result_table)

    score_table = get_table(
        'Scores', 'Module', 'Label', 'Score', 'Confidence', 'Weight'
    )
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


def display_test_result(
    result: pd.DataFrame, group_names: pd.DataFrame, model: DiscreteRrlModel
) -> None:
    result = result.merge(group_names, how='left', left_index=True, right_index=True)
    dataset_config, train_config = model.config.dataset, model.config.train
    # Only select data in the target groups
    X_df, y_df = make_data_labels(result, dataset_config, train_config)
    # Drop predictions that are failed
    y_df = y_df[~X_df.isna().all(axis=1)]
    X_df = X_df.dropna(how='all')
    y_true = y_df.map(train_config.get_group_index_mapping()).to_numpy()
    y_score = softmax(X_df.astype(float).values, axis=1)
    recorder = Recorder(train_config)
    recorder.record((0.0, y_true, y_score, {}))


def display_batched_result(config: RrlEvalConfig) -> None:
    df, result, group_names, model = get_batched_result(config)
    result_table = get_table('Result', 'ID', *df.columns)
    for row in df.itertuples():
        result_table.add_row(
            str(row.Index), *row[1:-2], f'{row.Score:.2f}', f'{row.Confidence:.2%}'
        )
    console.print(result_table)
    if config.debug and group_names is not None:
        display_test_result(result, group_names, model)


@app.command(sort_key=2)
def rrl_eval(*, config: RrlEvalConfig) -> None:
    """Evaluate an RRL model."""
    if config.batched:
        display_batched_result(config)
    else:
        display_result(config)
