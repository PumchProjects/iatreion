from cyclopts import App

from iatreion.configs import (
    RrlConfig,
    XgboostConfig,
)
from iatreion.trainers import ModelTrainer

from .common import app

sub_app = App(name='train', help='Train a model.')
app.command(sub_app)


@sub_app.command(sort_key=0)
def rrl(*, config: RrlConfig) -> None:
    """Train an RRL model."""
    from iatreion.trainers.rrl import RrlTrainer

    trainer = RrlTrainer(config)
    trainer.train()


@sub_app.command(sort_key=1)
def xgboost(*, config: XgboostConfig, **param) -> None:
    """Train an XGBoost model.

    Parameters
    ----------
    param: dict
        Parameters for XGBoost. See https://xgboost.readthedocs.io/en/stable/parameter.html
        for more details.
    """
    from iatreion.models import XgboostModel

    config.param = param
    model = XgboostModel(config)
    trainer = ModelTrainer(config.dataset, config.train, model)
    trainer.train()
