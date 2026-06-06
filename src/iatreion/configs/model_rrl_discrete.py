from dataclasses import dataclass

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig(ModelConfig):
    def __post_init__(self) -> None:
        self.dataset._encode = True
        if not self.train.final:
            self.register_log_dir(
                'rrl-discrete',
                folder_name=self.train.eval_name_str or None,
                file_name='test.log',
            )
