from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

from iatreion.utils import set_device, set_seed

from .dataset import DataName

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
    group_names: Annotated[
        list[str], Parameter(name=['--groups', '-g'], consume_multiple=True)
    ]
    'Group names of the data.'

    ref_names: Annotated[
        list[DataName] | None, Parameter(name=['--refs', '-r'], consume_multiple=True)
    ] = None
    """Names of the reference data files.
When final=False, align test data to the reference data.
Training data are also aligned when true_ref=True.
For discrete RRL, this parameter is used to gather the correct RRL models.
When final=True, the whole dataset is used for training and no alignment is performed.
For RRL, this parameter is used to extract the weight of the corresponding model.
When evaluating RRL, this parameter is useless.
"""

    true_ref: Annotated[bool, Parameter(name=['--true-ref', '-tr'])] = False
    'Align not only the test data, but also the training data to the reference data.'

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
        if not self.group_names:
            raise ValueError('No valid groups found.')
        for group in self.group_names:
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

    @property
    def group_name_str(self) -> str:
        return ', '.join(''.join(group) for group in self.groups)

    @property
    def ref_name_str(self) -> str:
        if self.final:
            return 'final'
        elif self.ref_names is None:
            return 'original'
        else:
            ref_names = ', '.join(self.ref_names)
            return f'{"ref" if self.true_ref else "of"} {ref_names}'

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
