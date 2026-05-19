from dataclasses import dataclass

from cyclopts import Parameter

from .model_base import FoldScope
from .show_result_base import ShowResultConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowInterpretabilityConfig(ShowResultConfig):
    fold_scope: FoldScope = 'outer'
    """Fold scope for aggregation.
'outer': use one fold per outer split.
'all': use all matched fold files.
"""

    top_k: int = 20
    'Number of top features to display in plots.'

    def __post_init__(self) -> None:
        super().__post_init__()
