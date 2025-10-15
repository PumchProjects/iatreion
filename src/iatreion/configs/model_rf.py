from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter

from .dataset import DatasetConfig
from .train import TrainConfig
from .utils import register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class RandomForestConfig:
    dataset: DatasetConfig

    train: TrainConfig

    n_estimators: int = 100
    'Number of trees in the forest.'

    n_jobs: int = 4
    'Number of jobs to run in parallel. Default is 4.'

    @property
    def score_file(self) -> Path:
        return self.train.log_dir / f'score_{self.train.ith_kfold}.json'

    def __post_init__(self) -> None:
        self.dataset.simple = True
        register_log_dir(self.dataset, self.train, 'random_forest')
