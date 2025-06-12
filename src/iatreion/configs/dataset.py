from dataclasses import dataclass
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory

type DataName = Literal[
    'biomarker',
    'cbf',
    'csvd',
    'volume',
    'snp',
    'tic-tac-toe',  # sanity test
]


@Parameter(name='*')
@dataclass(kw_only=True)
class DatasetConfig:
    prefix: Annotated[ExistingDirectory, Parameter(name=['--prefix', '-p'])]
    'Prefix of the data files.'

    name: Annotated[DataName, Parameter(name=['--name', '-n'])]
    'Name of the data file.'
