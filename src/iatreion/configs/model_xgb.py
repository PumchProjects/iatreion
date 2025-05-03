from dataclasses import dataclass, field
from typing import Annotated, Any

from cyclopts import Parameter

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
