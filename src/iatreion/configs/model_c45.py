from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter

from .model_base import ModelConfig

type C45ClassWeight = Literal['none', 'balanced']
type C45Criterion = Literal['entropy', 'log_loss']
type DecisionTreeSplitter = Literal['best', 'random']


@Parameter(name='*')
@dataclass(kw_only=True)
class C45Config(ModelConfig):
    criterion: C45Criterion = 'entropy'
    'Entropy-based split criterion for the C4.5-style tree.'

    splitter: DecisionTreeSplitter = 'best'
    'Strategy used to choose each split.'

    max_depth: int = 0
    'Maximum tree depth. Use 0 for unlimited depth.'

    min_samples_split: int = 2
    'Minimum number of samples required to split an internal node.'

    min_samples_leaf: int = 1
    'Minimum number of samples required at a leaf node.'

    max_features: str = 'none'
    'Number of features to consider at each split. Use sqrt, log2, none, or a numeric string.'

    class_weight: C45ClassWeight = 'none'
    'Class weighting strategy.'

    ccp_alpha: float = 0.0
    'Minimal cost-complexity pruning strength.'
