from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import Parameter

from .show_result_base import ShowResultConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowPerformanceConfig(ShowResultConfig):
    reference: Annotated[str | None, Parameter(alias='-ref')] = None
    'Reference label for comparison.'

    metrics: Annotated[list[str], Parameter(alias='-met', consume_multiple=True)] = (
        field(default_factory=lambda: ['AUC', 'SEN', 'SPC'])
    )
    'Metrics to include in the analysis.'

    def __post_init__(self) -> None:
        super().__post_init__()
