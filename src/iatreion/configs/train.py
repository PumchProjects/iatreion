from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import Directory

from iatreion.utils import set_device, set_seed


@Parameter(name='*')
@dataclass(kw_only=True)
class TrainConfig:
    group_names_: Annotated[str, Parameter(name=['--groups', '-g'])]
    'Group names of the data.'

    n_splits: Annotated[int, Parameter(name=['--n-splits', '-ns'])] = 10
    'Number of splits for cross-validation.'

    n_repeats: Annotated[int, Parameter(name=['--n-repeats', '-nr'])] = 10
    'Number of repeats for cross-validation.'

    device_id: Annotated[int, Parameter(name=['--device-id', '-i'])] = 0
    'Device ID for training. Default is 0.'

    final: Annotated[bool, Parameter(name=['--final', '-f'])] = False
    'Whether to use the whole dataset for training or testing.'

    seed: int = 42
    'Random seed for reproducibility.'

    plot_roc: bool = True
    'Plot ROC curve.'

    log_root: Directory = Path('logs')
    'Root directory for logs.'

    # TODO: why cannot use field(init=False) here?
    log_dir: Annotated[Directory, Parameter(parse=False)] = Path('logs')

    ith_kfold: Annotated[int, Parameter(parse=False)] = 0

    @cached_property
    def groups(self) -> list[list[str]]:
        groups: list[list[str]] = []
        for group in self.group_names_.split(','):
            names: list[str] = []
            i = 0
            while i < len(group):
                if group[i] in 'LEA':
                    names.append(group[i : i + 2])
                    i += 1
                else:
                    names.append(group[i])
                i += 1
            groups.append(sorted(names))
        return sorted(groups, key=lambda x: x[0])

    @cached_property
    def group_names(self) -> str:
        return ','.join(''.join(group) for group in self.groups)

    @property
    def n_folds(self) -> int:
        return self.n_splits * self.n_repeats

    @property
    def num_class(self) -> int:
        return len(self.groups)

    @property
    def label_pos(self) -> str:
        group_Ab = any(name in '12' for names in self.groups for name in names)
        return 'Ab' if group_Ab else 'encrypted'

    @property
    def roc_file(self) -> Path:
        return self.log_dir / 'roc.png'

    def __post_init__(self) -> None:
        set_device(self.device_id)
        set_seed(self.seed)
        if self.num_class > 2:
            # HACK: Disable ROC plot for multiclass classification
            self.plot_roc = False
