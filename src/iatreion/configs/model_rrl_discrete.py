from dataclasses import dataclass

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig(ModelConfig):
    def __post_init__(self) -> None:
        self.train._encode = True
        if not self.train.final:
            self.register_log_dir('rrl-discrete', file_name='test.log')
