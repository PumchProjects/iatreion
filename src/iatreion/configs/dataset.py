from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory

type DataName = Literal[
    'cdr',
    'mmse',
    'mmse-sum',
    'moca',
    'moca-sum',
    'adl',
    'biomarker',
    'cbf',
    'csvd',
    'volume',
    'volume-v',
    'volume-pct',
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

    place_holder: Annotated[str, Parameter(parse=False)] = 'Excalibur'

    @property
    def data(self) -> Path:
        return self.prefix / f'{self.name}.data'

    @property
    def info(self) -> Path:
        return self.prefix / f'{self.name}.info'

    @property
    def fmap(self) -> Path:
        return self.prefix / f'{self.name}.fmap'
