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
    'adl-sum',
    'screen-sum',
    'associative-learning',
    'episodic-memory',
    'avlt',
    's-composite-aea',
    'composite-bin',
    'biomarker',
    'cbf',
    'csvd',
    'volume',
    'volume-v',
    'volume-pct',
    'volume-v-nz',
    'volume-pct-nz',
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

    simple: bool = False
    'Whether to use the simple (non-binarized) version of the dataset.'

    place_holder: Annotated[str, Parameter(parse=False)] = 'Excalibur'

    @property
    def true_name(self) -> str:
        return f'{self.name}-simple' if self.simple else self.name

    @property
    def data(self) -> Path:
        return self.prefix / f'{self.true_name}.data'

    @property
    def info(self) -> Path:
        return self.prefix / f'{self.true_name}.info'

    @property
    def fmap(self) -> Path:
        return self.prefix / f'{self.true_name}.fmap'
