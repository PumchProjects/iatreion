from cyclopts import App

from iatreion.configs import RrlConfig, XgboostConfig
from iatreion.models import XGBoostModel
from iatreion.trainers import ModelTrainer, RrlTrainer

from .common import app

sub_app = App(name='train', help='Train a model.')
app.command(sub_app)


@sub_app.command(sort_key=0)
def rrl(*, config: RrlConfig) -> None:
    """Train an RRL model."""
    trainer = RrlTrainer(config)
    trainer.train()


@sub_app.command(sort_key=1)
def xgboost(*, config: XgboostConfig) -> None:
    """Train an XGBoost model."""
    model = XGBoostModel(config)
    trainer = ModelTrainer(config.dataset, config, config.train, model)
    trainer.train()
