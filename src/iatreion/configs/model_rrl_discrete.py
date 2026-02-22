from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

from .model_base import ModelConfig
from .utils import get_exp_root, register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig(ModelConfig):
    weight: Annotated[
        Literal['uniform', 'train-f1', 'val-f1', 'train-adaboost', 'val-adaboost'],
        Parameter(alias='-w'),
    ] = 'uniform'
    'Mode of model weight calculation.'

    def __post_init__(self) -> None:
        if not self.train.final:
            register_log_dir(
                self.dataset,
                self.train,
                'rrl-discrete',
                folder_name='val' if self.train.val_size is not None else self.weight,
                file_name='test.log',
            )

    def get_exp_roots(self) -> list[Path]:
        return [get_exp_root(name, self.train) for name in self.dataset.names]
