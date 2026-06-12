from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter

from .model_base import ModelConfig

type RandomForestClassWeight = Literal['none', 'balanced', 'balanced_subsample']
type RandomForestCriterion = Literal['gini', 'entropy', 'log_loss']


@Parameter(name='*')
@dataclass(kw_only=True)
class RandomForestConfig(ModelConfig):
    n_estimators: int = 100
    'Number of trees in the forest.'

    n_jobs: int = 4
    'Number of jobs to run in parallel. Default is 4.'

    criterion: RandomForestCriterion = 'gini'
    'Function used to measure split quality.'

    max_depth: int = 0
    'Maximum tree depth. Use 0 for unlimited depth.'

    min_samples_split: int = 2
    'Minimum number of samples required to split an internal node.'

    min_samples_leaf: int = 1
    'Minimum number of samples required at a leaf node.'

    max_features: str = 'sqrt'
    'Number of features to consider at each split. Use sqrt, log2, none, or a numeric string.'

    bootstrap: bool = True
    'Whether bootstrap samples are used when building trees.'

    class_weight: RandomForestClassWeight = 'none'
    'Class weighting strategy.'

    max_samples: float = 1.0
    'Fraction of samples drawn for each tree when bootstrap is enabled.'

    ccp_alpha: float = 0.0
    'Minimal cost-complexity pruning strength.'
