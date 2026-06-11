from dataclasses import dataclass, field
from typing import Annotated, Literal

from cyclopts import Parameter

from .dataset import DataName
from .model_base import ModelConfig

type SourceModelName = Literal['rrl', 'xgboost', 'random-forest']


@Parameter(name='*')
@dataclass(kw_only=True)
class ResultReplayConfig(ModelConfig):
    source_model: SourceModelName = 'rrl'
    'Model whose result NPZ files are replayed.'

    eval_names: Annotated[list[DataName], Parameter(consume_multiple=True)] = field(
        default_factory=list
    )
    'Subset of modalities to replay/fuse. Empty means all configured dataset names.'
