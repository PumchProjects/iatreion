from cyclopts import App

from iatreion.configs import GatreeConfig, GosdtConfig, RrlConfig, XgboostConfig
from iatreion.models import GatreeModel, GosdtModel, XgboostModel
from iatreion.trainers import ModelTrainer, RawModelTrainer

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
    config.param = param
    model = XgboostModel(config)
    trainer = ModelTrainer(config.dataset, config.train, model)
    trainer.train()


@sub_app.command(sort_key=2)
def gatree(*, config: GatreeConfig) -> None:
    """Train a GATree model."""
    model = GatreeModel(config)
    trainer = RawModelTrainer(config.dataset, config.train, model)
    trainer.train()


@sub_app.command(sort_key=3)
def gosdt(*, config: GosdtConfig) -> None:
    """Train a GOSDT model."""
    model = GosdtModel(config)
    trainer = RawModelTrainer(config.dataset, config.train, model)
    trainer.train()
