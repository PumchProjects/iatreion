from dataclasses import dataclass

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig(ModelConfig):
    @property
    def log_folder_name(self) -> str | None:
        if self.train.final:
            return self.dataset.name_str
        return self.train.eval_name_str or None

    def __post_init__(self) -> None:
        self.dataset._encode = True
