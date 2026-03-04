from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import Directory

from .dataset import DataName, DatasetConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowConfig:
    prefix: Annotated[Directory, Parameter(alias='-p')]
    'Prefix of the data files.'

    names: Annotated[list[DataName], Parameter(alias='-n', consume_multiple=True)]
    'Names of the data files.'

    groups: Annotated[list[str], Parameter(alias='-g', consume_multiple=True)]
    'Group names of the data.'

    root: Directory = Path('figures')
    'Root directory for figures and tables.'

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def make_configs(self) -> tuple[DatasetConfig, TrainConfig]:
        dataset_config = DatasetConfig(prefix=self.prefix, names=self.names)
        train_config = TrainConfig(group_names=self.groups, _shuffle=False)
        return dataset_config, train_config
