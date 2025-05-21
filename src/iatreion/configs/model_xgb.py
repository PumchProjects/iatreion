from dataclasses import dataclass, field
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

    num_round: Annotated[int, Parameter(name=['--num_round', '-nr'])] = 100
    'Set the number of boosting rounds.'

    def __post_init__(self) -> None:
        self.train.log_dir = (
            self.train.log_root / self.dataset.name / self.train.groups / 'xgboost'
        )
        add_file_handler(self.train.log_dir / 'train.log')
