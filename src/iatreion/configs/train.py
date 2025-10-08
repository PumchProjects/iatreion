from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

from iatreion.utils import set_device, set_seed

type SamplerName = Literal[
    'adasyn',
    'smote',
    'smotetomek',
    'smoteenn',
    'borderlinesmote-1',
    'borderlinesmote-2',
    'svmsmote',
    'kmeanssmote',
]


@Parameter(name='*')
@dataclass(kw_only=True)
class TrainConfig:
    group_names: Annotated[str, Parameter(name=['--groups', '-g'])]
    'Group names of the data.'

    n_splits: Annotated[int, Parameter(name=['--n-splits', '-ns'])] = 10
    'Number of splits for cross-validation.'

    n_repeats: Annotated[int, Parameter(name=['--n-repeats', '-nr'])] = 10
    'Number of repeats for cross-validation.'

    device_id: Annotated[int, Parameter(name=['--device-id', '-i'])] = 0
    'Device ID for training. Default is 0.'

    final: Annotated[bool, Parameter(name=['--final', '-f'])] = False
    'Whether to use the whole dataset for training or testing.'

    over_sampler: Annotated[
        SamplerName | None, Parameter(name=['--over-sampler', '-os'])
    ] = None
    'Over-sampling method to use.'

    min_n_samples: Annotated[int, Parameter(name=['--min-n-samples', '-mns'])] = 0
    'Minimum number of samples for each class after resampling.'

    seed: int = 42
    'Random seed for reproducibility.'

    plot_roc: bool = True
    'Plot ROC curve.'

    log_root: Directory = Path('logs')
    'Root directory for logs.'

    # TODO: why cannot use field(init=False) here?
    log_dir: Annotated[Directory, Parameter(parse=False)] = Path('logs')

    ith_kfold: Annotated[int, Parameter(parse=False)] = 0

    base_pos: Annotated[str, Parameter(parse=False)] = ''

    label_pos: Annotated[str, Parameter(parse=False)] = 'encrypted'

    groups: Annotated[list[list[str]], Parameter(parse=False)] = field(
        default_factory=list
    )

    def set_groups(self) -> None:
        if self.group_names.strip() == '':
            raise ValueError('No valid groups found.')
        for group in self.group_names.split(','):
            names: list[str] = []
            i = 0
            while i < len(group):
                if group[i].isupper():
                    if group[i + 1] in ['<', '>']:
                        self.base_pos = 'AC 60'
                        names.append(group[i : i + 4])
                        i += 4
                    else:
                        self.base_pos = 'AC to 3'
                        names.append(group[i : i + 2])
                        i += 2
                else:
                    if group[i] in '12':
                        self.label_pos = 'Ab'
                    names.append(group[i])
                    i += 1
            self.groups.append(sorted(names))
        self.groups.sort(key=lambda x: x[0])
        self.group_names = ','.join(''.join(group) for group in self.groups)

    @property
    def n_folds(self) -> int:
        return self.n_splits * self.n_repeats

    @property
    def num_class(self) -> int:
        return len(self.groups)

    @property
    def roc_file(self) -> Path:
        return self.log_dir / 'roc.png'

    def __post_init__(self) -> None:
        set_device(self.device_id)
        set_seed(self.seed)
        if self.num_class > 2:
            # HACK: Disable ROC plot for multiclass classification
            self.plot_roc = False
        self.set_groups()
