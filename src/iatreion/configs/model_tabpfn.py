from dataclasses import dataclass

from cyclopts import Parameter
from cyclopts.types import ExistingFile

from .baseline_eval import BaselineEvalConfig
from .model_base import ModelConfig
from .preprocessor import PreprocessorConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class TabPFNConfig(ModelConfig):
    model_path: ExistingFile
    'Path to the pretrained TabPFN model file.'

    n_jobs: int = 4
    'Number of worker processes to use for the preprocessing. Default is 4.'


@Parameter(name='*')
@dataclass(kw_only=True)
class TabPFNEvalConfig(BaselineEvalConfig):
    model_path: ExistingFile
    'Path to the pretrained TabPFN model file.'

    def make_configs(
        self, model_config_cls: type[ModelConfig]
    ) -> tuple[PreprocessorConfig, ModelConfig]:
        return self._make_configs(model_config_cls, model_path=self.model_path)
