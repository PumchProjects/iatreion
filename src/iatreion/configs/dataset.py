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
]


@Parameter(name='*')
@dataclass(kw_only=True)
class DatasetConfig:
    prefix: Annotated[ExistingDirectory, Parameter(name=['--prefix', '-p'])]
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

    group_columns: Annotated[list[str], Parameter(parse=False)] = field(
        default_factory=lambda: ['encrypted', 'Ab', 'AC to 3', 'AC 60']
    )

    index_name: Annotated[str, Parameter(parse=False)] = 'serial_num'

    place_holder: Annotated[str, Parameter(parse=False)] = 'Excalibur'

    @property
    def name_str(self) -> str:
        return ', '.join(self.names)

    def true_name(self, name: DataName) -> str:
        return f'{name}-simple' if self.simple else name

    def get_data(self, name: DataName, prefix: Path | None = None) -> Path:
        return (prefix or self.prefix) / f'{self.true_name(name)}.data'

    def get_info(self, name: DataName, prefix: Path | None = None) -> Path:
        return (prefix or self.prefix) / f'{self.true_name(name)}.info'

    def get_fmap(self, name: DataName, prefix: Path | None = None) -> Path:
        return (prefix or self.prefix) / f'{self.true_name(name)}.fmap'
