from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig(ModelConfig):
    _weight: Annotated[
        Literal['uniform', 'train-f1', 'val-f1', 'train-adaboost', 'val-adaboost'],
        Parameter(alias='-w'),
    ] = 'uniform'
    'Mode of model weight calculation.'

    def __post_init__(self) -> None:
        if not self.train.final:
            self.register_log_dir('rrl-discrete', file_name='test.log')
