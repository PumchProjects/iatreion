from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter

from .model_base import ModelConfig

type XgboostDevice = Literal['cpu', 'cuda']
type XgboostTreeMethod = Literal['auto', 'exact', 'approx', 'hist']


@Parameter(name='*')
@dataclass(kw_only=True)
class XgboostConfig(ModelConfig):
    num_round: int = 100
    'Set the number of boosting rounds.'

    device: XgboostDevice = 'cpu'
    'Device for XGBoost training.'

    tree_method: XgboostTreeMethod = 'auto'
    'Tree construction algorithm.'

    learning_rate: float = 0.3
    'Boosting learning rate.'

    max_depth: int = 6
    'Maximum tree depth.'

    min_child_weight: float = 1.0
    'Minimum Hessian sum required in a child node.'

    subsample: float = 1.0
    'Subsample ratio of training rows for each tree.'

    colsample_bytree: float = 1.0
    'Subsample ratio of columns for each tree.'

    gamma: float = 0.0
    'Minimum loss reduction required to split a node.'

    reg_lambda: float = 1.0
    'L2 regularization term on leaf weights.'

    reg_alpha: float = 0.0
    'L1 regularization term on leaf weights.'

    scale_pos_weight: float = 1.0
    'Positive-class weight multiplier for binary classification.'
