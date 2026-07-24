from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter

from .model_base import ModelConfig

type LogisticRegressionClassWeight = Literal['none', 'balanced']
type LogisticRegressionPenalty = Literal['none', 'l2']
type LogisticRegressionSolver = Literal['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']


@Parameter(name='*')
@dataclass(kw_only=True)
class LogisticRegressionConfig(ModelConfig):
    penalty: LogisticRegressionPenalty = 'l2'
    'Regularization penalty. Use none to disable regularization.'

    C: float = 1.0
    'Inverse regularization strength.'

    solver: LogisticRegressionSolver = 'lbfgs'
    'Optimization algorithm.'

    max_iter: int = 5000
    'Maximum number of optimizer iterations.'

    tol: float = 1e-4
    'Stopping tolerance.'

    class_weight: LogisticRegressionClassWeight = 'none'
    'Class weighting strategy.'
