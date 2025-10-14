from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from cyclopts import Parameter

from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class XgboostConfig:
    dataset: DatasetConfig

    train: TrainConfig

    param: Annotated[dict[str, Any], Parameter(parse=False)] = field(
        default_factory=dict
    )

    num_round: int = 100
    'Set the number of boosting rounds.'

    @property
    def score_file(self) -> Path:
        return self.train.log_dir / f'score_{self.train.ith_kfold}.json'

    def __post_init__(self) -> None:
        self.dataset.simple = True
        self.train.log_dir = (
            self.train.log_root
            / self.dataset.name_str
            / self.train.group_name_str
            / 'xgboost'
            / self.train.ref_name_str
        )
        add_file_handler(self.train.log_dir / 'train.log')
