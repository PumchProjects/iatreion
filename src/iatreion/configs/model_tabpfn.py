from dataclasses import dataclass

from cyclopts import Parameter
from cyclopts.types import ExistingFile

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class TabPFNConfig(ModelConfig):
    model_path: ExistingFile
    'Path to the pretrained TabPFN model file.'

    calc_importance: bool = False
    'Whether to calculate feature importance.'

    n_jobs: int = 4
    'Number of worker processes to use for the preprocessing. Default is 4.'
