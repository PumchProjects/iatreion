from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from cyclopts import Parameter

from .dataset import DatasetConfig
from .train import TrainConfig
from .utils import register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class XgboostConfig:
    dataset: DatasetConfig

    train: TrainConfig

    param: Annotated[dict[str, Any], Parameter(parse=False)] = field(
        default_factory=dict[str, Any]
    )

    num_round: int = 100
    'Set the number of boosting rounds.'

    @property
    def score_file(self) -> Path:
        return self.train.log_dir / f'score_{self.train.ith_kfold}.json'

    def __post_init__(self) -> None:
        self.dataset.simple = True
        register_log_dir(self.dataset, self.train, 'xgboost')
