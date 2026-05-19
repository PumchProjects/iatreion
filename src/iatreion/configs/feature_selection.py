from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import PositiveFloat, PositiveInt
from cyclopts.validators import Number

type FeatureSelectionMethod = Literal[
    'none',
    'f_classif',
    'mutual_info',
    'l1_logistic',
    'auc',
    'logistic_lrt',
]
type FeatureSelectionScoreAggregate = Literal['max', 'mean']


def name_transform(s: str) -> str:
    return f'feature-selection-{s.replace("_", "-")}'


@Parameter(name='*', name_transform=name_transform)
@dataclass(kw_only=True)
class FeatureSelectionConfig:
    method: FeatureSelectionMethod = 'none'
    """Supervised feature-selection method fitted inside each training fold.
'none': keep all features.
'f_classif': ANOVA F-test, supports binary and multiclass labels.
'mutual_info': mutual information classifier score, supports binary and multiclass labels.
'l1_logistic': multinomial L1 logistic-regression embedded selection.
'auc': binary-only univariate AUROC distance from chance.
'logistic_lrt': binary-only univariate logistic likelihood-ratio score.
"""

    fraction: Annotated[float, Parameter(validator=Number(gt=0, lte=1))] = 0.5
    'Fraction of raw features to keep when supervised feature selection is enabled.'

    top_k: PositiveInt | None = None
    'Exact number of raw features to keep. Overrides fraction when set.'

    min_features: PositiveInt = 1
    'Minimum number of raw features to keep when supervised feature selection is enabled.'

    max_features: PositiveInt | None = None
    'Maximum number of raw features to keep when supervised feature selection is enabled.'

    score_aggregate: FeatureSelectionScoreAggregate = 'max'
    'How to aggregate category-level scores back to one raw unordered feature score.'

    c: PositiveFloat = 1.0
    'Inverse regularization strength for l1_logistic feature selection.'

    def validate(self) -> None:
        if self.max_features is None:
            return
        if self.min_features > self.max_features:
            raise ValueError(
                'feature_selection.min_features cannot be greater than '
                'feature_selection.max_features.'
            )
