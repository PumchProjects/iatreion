from itertools import count
from pathlib import Path

from cyclopts import App

from iatreion.api import (
    get_baseline_batched_result,
    get_baseline_eval_result,
    save_baseline_batched_result_table,
)
from iatreion.configs import (
    BaselineEvalConfig,
    ModelConfig,
    RandomForestConfig,
    RrlEvalConfig,
    XgboostConfig,
)
from iatreion.models import Model, RandomForestModel, XgboostModel
from iatreion.models.naming import model_name_for
from iatreion.utils import logger

from .common import console
from .eval_rrl import run_rrl_eval

sub_app = App(name='eval')
counter = count()


def display_batched_result(
    config: BaselineEvalConfig,
    model_cls: type[Model],
    model_config_cls: type[ModelConfig],
) -> None:
    table, model_config = get_baseline_batched_result(
        config, model_cls, model_config_cls
    )
    try:
        output = save_baseline_batched_result_table(
            table, Path(config.output or 'baseline_batch_result.xlsx')
        )
        console.print(f'Saved baseline batch result to {output}')
    finally:
        model_config.close_log_handler()


def display_eval_result(
    config: BaselineEvalConfig,
    model_cls: type[Model],
    model_config_cls: type[ModelConfig],
) -> None:
    result, fig, model_config = get_baseline_eval_result(
        config, model_cls, model_config_cls
    )
    try:
        model_config.register_eval_log_dir(model_name_for(model_cls))
        logger.info(result)
        if fig is not None:
            dataset, train = model_config.dataset, model_config.train
            fig.savefig(train.get_roc_file(dataset.name_str), dpi=300)
    finally:
        model_config.close_log_handler()


def evaluate_baseline(
    config: BaselineEvalConfig,
    model_cls: type[Model],
    model_config_cls: type[ModelConfig],
) -> None:
    match config.mode:
        case 'batch':
            display_batched_result(config, model_cls, model_config_cls)
        case 'eval':
            display_eval_result(config, model_cls, model_config_cls)


@sub_app.command(sort_key=next(counter))
def rrl(*, config: RrlEvalConfig | None = None) -> None:
    """Evaluate final RRL rule files with the parser."""
    if config is None:
        config = RrlEvalConfig()
    run_rrl_eval(config)


@sub_app.command(sort_key=next(counter))
def xgboost(*, config: BaselineEvalConfig | None = None) -> None:
    """Evaluate final XGBoost models."""
    if config is None:
        config = BaselineEvalConfig()
    evaluate_baseline(config, XgboostModel, XgboostConfig)


@sub_app.command(sort_key=next(counter))
def random_forest(*, config: BaselineEvalConfig | None = None) -> None:
    """Evaluate final Random Forest models."""
    if config is None:
        config = BaselineEvalConfig()
    evaluate_baseline(config, RandomForestModel, RandomForestConfig)
