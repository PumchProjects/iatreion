from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import Directory

from .dataset import DataName


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowConfig:
    names: Annotated[list[DataName], Parameter(alias='-n', consume_multiple=True)]
    'Names of the data files.'

    groups: Annotated[list[str], Parameter(alias='-g', consume_multiple=True)]
    'Group names of the data.'

    title: Annotated[str, Parameter(alias='-t')] = ''
    'Title for the figure or table to show.'

    output: Annotated[str, Parameter(alias='-o')]
    'Output file name for the figure or table to show.'

    root: Directory = Path('figures')
    'Root directory for figures and tables.'

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, suffix: str) -> Path:
        return self.root / f'{self.output}.{suffix}'
