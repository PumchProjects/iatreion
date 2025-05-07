from dataclasses import dataclass

from cyclopts import Parameter

from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class GatreeConfig:
    dataset: DatasetConfig

    train: TrainConfig

    n_jobs: int = 16
    'Set the number of parallel processes to use for training and prediction.'

    population_size: int = 100
    'Set the population size for GATree.'

    max_iter: int = 100
    'Set the maximum number of iterations for GATree.'

    def __post_init__(self) -> None:
        self.train.log_dir = (
            self.train.log_root / self.dataset.name / self.train.groups / 'gatree'
        )
        add_file_handler(self.train.log_dir / 'log.txt')
        self.train.record_auc = False
