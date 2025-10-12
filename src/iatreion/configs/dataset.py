from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory

type DataName = Literal[
    'life',
    'diet-medication',
    'family-history',
    'medical-history',
    'symptom',
    's-history',
    'cdr',
    'mmse',
    'mmse-sum',
    'moca',
    'moca-sum',
    'adl',
    'adl-sum',
    's-screen-sum',
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
    's-all',
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

    group_columns: Annotated[list[str], Parameter(parse=False)] = field(
        default_factory=lambda: ['encrypted', 'Ab', 'AC to 3', 'AC 60']
    )

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
