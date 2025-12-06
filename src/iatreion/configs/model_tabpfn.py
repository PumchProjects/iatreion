from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingFile

from .dataset import DatasetConfig
from .train import TrainConfig
from .utils import register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class TabPFNConfig:
    dataset: DatasetConfig

    train: TrainConfig

    model_path: Annotated[ExistingFile, Parameter(name=['--model-path', '-mp'])]
    'Path to the pretrained TabPFN model file.'

    calc_importance: Annotated[bool, Parameter(name=['--calc-importance', '-ci'])] = (
        False
    )
    'Whether to calculate feature importance.'

    n_jobs: int = 4
    'Number of worker processes to use for the preprocessing. Default is 4.'

    @property
    def score_file(self) -> Path:
        return self.train.log_dir / f'score_{self.train.ith_kfold}.json'

    def __post_init__(self) -> None:
        self.dataset.simple = True
        register_log_dir(self.dataset, self.train, 'tabpfn')
