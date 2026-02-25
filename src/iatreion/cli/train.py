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

sub_app = App(name='train', help='Train a model.', sort_key=1)
counter = count()


def train(config: ModelConfig, model: Model) -> None:
    ModelTrainer(config, model).train()


@sub_app.command(sort_key=next(counter))
def rrl(*, config: RrlConfig) -> None:
    """Train an RRL model."""
    from iatreion.trainers.rrl import RrlTrainer

    RrlTrainer(config).train()


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
    train(config, XgboostModel(config))


@sub_app.command(sort_key=next(counter))
def random_forest(*, config: RandomForestConfig) -> None:
    """Train a Random Forest model."""
    train(config, RandomForestModel(config))


@sub_app.command(sort_key=next(counter))
def tabpfn(*, config: TabPFNConfig) -> None:
    """Train a TabPFN model."""
    from iatreion.models.tabpfn import TabPFNModel

    train(config, TabPFNModel(config))


@sub_app.command(sort_key=next(counter))
def limix(*, config: LimiXConfig) -> None:
    """Train a LimiX model."""
    train(config, LimiXModel(config))


@sub_app.command(sort_key=next(counter))
def rrl_eval(*, config: DiscreteRrlConfig) -> None:
    """Evaluate trained RRL models."""
    train(config, DiscreteRrlModel(config))
