from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class LimiXConfig(ModelConfig):
    @property
    def inference_config_path(self) -> Path:
        return self.train.limix_repo_path / 'config' / 'cls_default_noretrieval.json'

    def __post_init__(self) -> None:
        self.register_log_dir('limix')
