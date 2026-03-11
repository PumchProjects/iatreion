from dataclasses import dataclass
from typing import Annotated

from cyclopts import Parameter

from .model_base import FoldScope
from .show_result_base import ShowResultConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowInterpretabilityConfig(ShowResultConfig):
    fold_scope: Annotated[FoldScope, Parameter(alias='-fs')] = 'outer'
    """Fold scope for aggregation.
'outer': use one fold per outer split.
'all': use all matched fold files.
"""

    top_k: Annotated[int, Parameter(alias='-top')] = 20
    'Number of top features to display in plots.'

    def __post_init__(self) -> None:
        super().__post_init__()
