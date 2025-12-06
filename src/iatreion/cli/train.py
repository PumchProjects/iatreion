from typing import Any

from cyclopts import App

from iatreion.configs import (
    DiscreteRrlConfig,
    RandomForestConfig,
    RrlConfig,
    TabPFNConfig,
    XgboostConfig,
)
from iatreion.models import DiscreteRrlModel, RandomForestModel, XgboostModel
from iatreion.trainers import ModelTrainer, RawModelTrainer

from .common import app

sub_app = App(name='train', help='Train a model.', sort_key=1)
app.command(sub_app)


@sub_app.command(sort_key=0)
def rrl(*, config: RrlConfig) -> None:
    """Train an RRL model."""
    from iatreion.trainers.rrl import RrlTrainer

    trainer = RrlTrainer(config)
    trainer.train()


@sub_app.command(sort_key=1)
def xgboost(*, config: XgboostConfig, **param: Any) -> None:
    """Train an XGBoost model.

    Parameters
    ----------
    param: dict
        Parameters for XGBoost. See https://xgboost.readthedocs.io/en/stable/parameter.html
        for more details.
    """
    config.param = param
    model = XgboostModel(config)
    trainer = ModelTrainer(config.dataset, config.train, model)
    trainer.train()


@sub_app.command(sort_key=2)
def random_forest(*, config: RandomForestConfig) -> None:
    """Train a Random Forest model."""
    model = RandomForestModel(config)
    trainer = ModelTrainer(config.dataset, config.train, model)
    trainer.train()


@sub_app.command(sort_key=3)
def tabpfn(*, config: TabPFNConfig) -> None:
    """Train a TabPFN model."""
    from iatreion.models.tabpfn import TabPFNModel

    model = TabPFNModel(config)
    trainer = ModelTrainer(config.dataset, config.train, model)
    trainer.train()


@sub_app.command(sort_key=4)
def rrl_eval(*, config: DiscreteRrlConfig) -> None:
    """Evaluate trained RRL models."""
    model = DiscreteRrlModel(config)
    trainer = RawModelTrainer(config.dataset, config.train, model)
    trainer.train()
