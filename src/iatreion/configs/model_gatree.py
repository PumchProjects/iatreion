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

    max_depth: int | None = None
    'The maximum depth of the tree.'

    n_jobs: int = 20
    'Set the number of parallel processes to use for training and prediction.'

    population_size: int = 100
    'Set the population size for GATree.'

    max_iter: int = 100
    'Set the maximum number of iterations for GATree.'

    mutation_probability: float = 0.1
    'Set the mutation probability for GATree.'

    elite_size: int = 1
    'Set the elite size for GATree.'

    selection_tournament_size: int = 2
    'Set the selection tournament size for GATree.'

    plot: bool = False
    'Whether to plot the tree.'

    def __post_init__(self) -> None:
        self.train.log_dir = (
            self.train.log_root / self.dataset.name / self.train.groups / 'gatree'
        )
        add_file_handler(self.train.log_dir / 'log.txt')
        self.train.record_auc = False
