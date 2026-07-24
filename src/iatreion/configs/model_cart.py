from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter

from .model_base import ModelConfig
from .model_c45 import DecisionTreeSplitter

type CartClassWeight = Literal['none', 'balanced']


@Parameter(name='*')
@dataclass(kw_only=True)
class CartConfig(ModelConfig):
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

    class_weight: CartClassWeight = 'none'
    'Class weighting strategy.'

    ccp_alpha: float = 0.0
    'Minimal cost-complexity pruning strength.'
