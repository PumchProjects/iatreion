from dataclasses import dataclass
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import Directory

from .dataset import DatasetConfig
from .show_base import ShowConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowDataConfig(ShowConfig):
    prefix: Annotated[Directory, Parameter(alias='-p')]
    'Prefix of the data files.'

    def make_configs(self) -> tuple[DatasetConfig, TrainConfig]:
        dataset_config = DatasetConfig(prefix=self.prefix, names=self.names)
        train_config = TrainConfig(group_names=self.groups, _shuffle=False)
        return dataset_config, train_config
