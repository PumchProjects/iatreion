from cyclopts import App

from iatreion.configs import RrlConfig
from iatreion.trainers import RrlTrainer

from .common import app

sub_app = App(name='train', help='Train a model.')
app.command(sub_app)


@sub_app.command(sort_key=0)
def rrl(*, config: RrlConfig) -> None:
    """Train an RRL model."""
    trainer = RrlTrainer(config)
    trainer.train()
