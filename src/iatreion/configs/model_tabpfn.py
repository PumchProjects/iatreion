from dataclasses import dataclass
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingFile

from .model_base import ModelConfig
from .utils import register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class TabPFNConfig(ModelConfig):
    model_path: Annotated[ExistingFile, Parameter(alias='-mp')]
    'Path to the pretrained TabPFN model file.'

    calc_importance: Annotated[bool, Parameter(alias='-ci', negative='')] = False
    'Whether to calculate feature importance.'

    n_jobs: int = 4
    'Number of worker processes to use for the preprocessing. Default is 4.'

    def __post_init__(self) -> None:
        register_log_dir(self.dataset, self.train, 'tabpfn')
