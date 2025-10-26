from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

type DataName = Literal[
    'basic-noage',
    'basic',
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
    'had',
    'had-sum',
    's-screen-noage-sum',
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
    'volume-z-v',
    'volume-z-pct',
    'volume-nz-v',
    'volume-nz-pct',
    'volume-new-v',
    'volume-new-pct',
    'snp',
    'test-mmse-sum',
    'test-moca-sum',
    'test-adl-sum',
    'test-had-sum',
    'test-s-screen-sum',
    'test-volume-z-pct',
    'test-s-all',
]


@Parameter(name='*')
@dataclass(kw_only=True)
class DatasetConfig:
    prefix: Annotated[Directory, Parameter(name=['--prefix', '-p'])]
    'Prefix of the data files.'

    names: Annotated[
        list[DataName], Parameter(name=['--names', '-n'], consume_multiple=True)
    ]
    """Names of the data files.
For discrete RRL, separate models are evaluated and then aggregated.
For other models, features are concatenated.
"""

    simple: Annotated[bool, Parameter(parse=False)] = False
    'Whether to use the simple (non-binarized) version of the dataset.'

    @property
    def name_str(self) -> str:
        return ', '.join(self.names)

    def get_true_name(self, name: DataName) -> str:
        return f'{name}-simple' if self.simple else name

    def get_data(self, name: DataName) -> Path:
        return self.prefix / f'{self.get_true_name(name)}.data'

    def get_info(self, name: DataName) -> Path:
        return self.prefix / f'{self.get_true_name(name)}.info'

    def get_fmap(self, name: DataName) -> Path:
        return self.prefix / f'{self.get_true_name(name)}.fmap'
