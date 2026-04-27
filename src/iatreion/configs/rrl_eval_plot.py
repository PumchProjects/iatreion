from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

from .rrl_eval import RrlEvalConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class RrlEvalPlotConfig(RrlEvalConfig):
    mode: Literal['plot'] = field(init=False, default='plot')
    'Mode of RRL evaluation.'

    top_k: Annotated[int, Parameter(alias='-top')] = 20
    'Number of active rules to display per module.'

    title: Annotated[str, Parameter(alias='-pt')] = ''
    'Optional title for the generated figure.'

    output: Annotated[str, Parameter(alias='-o')] = 'rrl_waterfall'
    'Output file name for the generated figure and table.'

    root: Directory = Path('figures')
    'Root directory for generated figures and tables.'

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, suffix: str) -> Path:
        return self.root / f'{self.output}.{suffix}'
