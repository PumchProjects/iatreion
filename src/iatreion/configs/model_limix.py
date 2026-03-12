from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory, ExistingFile

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class LimiXConfig(ModelConfig):
    python_path: ExistingFile
    'Path to the Python interpreter that supports LimiX.'

    repo_path: ExistingDirectory
    'Path to the LimiX repository.'

    model_path: ExistingFile
    'Path to the pre-trained LimiX model file.'

    @property
    def inference_config_path(self) -> Path:
        return self.repo_path / 'config' / 'cls_default_noretrieval.json'

    def __post_init__(self) -> None:
        self.register_log_dir('limix')
