from dataclasses import dataclass
from typing import Annotated

from cyclopts import Parameter


@Parameter(name='*')
@dataclass(kw_only=True)
class TrainConfig:
    groups: Annotated[str, Parameter(name=['--groups', '-g'])]
    'Group names of the data.'

    n_splits: Annotated[int, Parameter(name=['--n-splits', '-ns'])] = 5
    'Number of splits for cross-validation.'

    @property
    def num_class(self) -> int:
        return len(self.groups)

    @property
    def label_pos(self) -> str:
        group_Ab = '1' in self.groups or '2' in self.groups
        return 'Ab' if group_Ab else 'encrypted'
