from dataclasses import dataclass

from cyclopts import Parameter

from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class CartConfig:
    dataset: DatasetConfig

    train: TrainConfig

    max_depth: int | None = None
    'The maximum depth of the tree.'

    plot: bool = False
    'Whether to plot the tree.'

    def __post_init__(self) -> None:
        self.train.log_dir = (
            self.train.log_root / self.dataset.name / self.train.groups / 'cart'
        )
        add_file_handler(self.train.log_dir / 'log.txt')
        self.train.record_auc = False
