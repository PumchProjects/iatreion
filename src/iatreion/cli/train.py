from itertools import count
from typing import Any

from cyclopts import App

from iatreion.configs import (
    DiscreteRrlConfig,
    LimiXConfig,
    ModelConfig,
    RandomForestConfig,
    RrlConfig,
    TabPFNConfig,
    XgboostConfig,
)
from iatreion.models import (
    DiscreteRrlModel,
    LimiXModel,
    Model,
    RandomForestModel,
    XgboostModel,
)
from iatreion.trainers import ModelTrainer
from iatreion.tuning import OptunaRunner
from iatreion.utils import progress

sub_app = App(name='train', help='Train a model.', sort_key=1)
counter = count()


def train(model_cls: type[Model], config: ModelConfig) -> None:
    with progress:
        if config.tune:
            try:
                OptunaRunner(model_cls, config).run()
            finally:
                config.close_log_handler()
            return

        model = model_cls(config)
        try:
            ModelTrainer(model).train()
        finally:
            model.close()
            config.close_log_handler()


@sub_app.command(sort_key=next(counter))
def rrl(*, config: RrlConfig) -> None:
    """Train an RRL model."""
    from iatreion.models.rrl import RrlModel

    train(RrlModel, config)


@sub_app.command(sort_key=next(counter))
def xgboost(*, config: XgboostConfig, **param: Any) -> None:
    """Train an XGBoost model.

    Parameters
    ----------
    param: dict
        Parameters for XGBoost. See https://xgboost.readthedocs.io/en/stable/parameter.html
        for more details.
    """
    config._param = param
    train(XgboostModel, config)


@sub_app.command(sort_key=next(counter))
def random_forest(*, config: RandomForestConfig) -> None:
    """Train a Random Forest model."""
    train(RandomForestModel, config)


@sub_app.command(sort_key=next(counter))
def tabpfn(*, config: TabPFNConfig) -> None:
    """Train a TabPFN model."""
    from iatreion.models.tabpfn import TabPFNModel

    train(TabPFNModel, config)


@sub_app.command(sort_key=next(counter))
def limix(*, config: LimiXConfig) -> None:
    """Train a LimiX model."""
    train(LimiXModel, config)


@sub_app.command(sort_key=next(counter))
def rrl_eval(*, config: DiscreteRrlConfig) -> None:
    """Evaluate trained RRL models."""
    train(DiscreteRrlModel, config)
