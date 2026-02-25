from dataclasses import dataclass, field
from typing import Any

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class XgboostConfig(ModelConfig):
    _param: dict[str, Any] = field(default_factory=dict[str, Any])

    num_round: int = 100
    'Set the number of boosting rounds.'

    def __post_init__(self) -> None:
        self.register_log_dir('xgboost')
