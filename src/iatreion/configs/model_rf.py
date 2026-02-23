from dataclasses import dataclass

from cyclopts import Parameter

from .model_base import ModelConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class RandomForestConfig(ModelConfig):
    n_estimators: int = 100
    'Number of trees in the forest.'

    n_jobs: int = 4
    'Number of jobs to run in parallel. Default is 4.'

    def __post_init__(self) -> None:
        self.register_log_dir('random_forest')
