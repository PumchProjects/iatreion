from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

from iatreion.utils import expand_range, set_device, set_seed

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

    keep: Annotated[Literal['all', 'first', 'last'], Parameter(alias='-k')] = 'all'
    """Deduplication strategy for duplicated samples.
'first': keep the first sample of each patient.
'last': keep the last sample of each patient.
'all' (default): do not deduplicate samples.
"""

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

    true_ref: Annotated[bool, Parameter(alias='-tr', negative='')] = False
    'Align not only the test data, but also the training data to the reference data.'

    n_splits: Annotated[int, Parameter(alias='-ns')] = 10
    'Number of splits for cross-validation.'

    n_repeats: Annotated[int, Parameter(alias='-nr')] = 10
    'Number of repeats for cross-validation.'

    device_id: Annotated[int, Parameter(alias='-i')] = 0
    'Device ID for training. Default is 0.'

    final: Annotated[bool, Parameter(alias='-f', negative='')] = False
    'Whether to use the whole dataset for training or testing.'

    over_sampler: Annotated[SamplerName | None, Parameter(alias='-os')] = None
    'Over-sampling method to use.'

    min_n_samples: Annotated[int, Parameter(alias='-mns')] = 0
    'Minimum number of samples for each class after resampling.'

    val_size: Annotated[float | int | None, Parameter(alias='-vs')] = None
    """If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
If int, represents the absolute number of validation samples.
If None (default), no validation set is used.
For discrete RRL, validation set is used for optimization when val_size is set.
"""

    suspected_case: Annotated[bool, Parameter(alias='-sc', negative='')] = False
    'Whether to include suspected cases in training.'

    label_name: Annotated[str | None, Parameter(alias='-ln')] = None
    'Label column name in the data files. If not set, determined automatically.'

    seed: int = 42
    'Random seed for reproducibility.'

    plot_roc: Annotated[bool, Parameter(negative='')] = True
    'Plot ROC curve.'

    log_root: Directory = Path('logs')
    'Root directory for logs.'

    # TODO: why cannot use field(init=False) here?
    log_dir: Annotated[Directory, Parameter(parse=False)] = Path('logs')

    ith_kfold: Annotated[int, Parameter(parse=False)] = 0

    base_pos: Annotated[str, Parameter(parse=False)] = ''

    label_pos: Annotated[str, Parameter(parse=False)] = 'group_encrypted'

    groups: Annotated[list[list[str]], Parameter(parse=False)] = field(
        default_factory=list[list[str]]
    )

    def set_groups(self) -> None:
        if not self.group_names:
            raise ValueError('No valid groups found.')
        for group in self.group_names:
            group = expand_range(group)
            names: list[str] = []
            i = 0
            while i < len(group):
                if group[i].isupper() and i + 1 < len(group):
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
                        self.label_pos = 'group_Ab'
                    names.append(group[i])
                    i += 1
            self.groups.append(sorted(names))
        self.groups.sort(key=lambda x: x[0])
        if self.label_name is not None:
            self.base_pos = ''
            self.label_pos = self.label_name

    def get_name_group_mapping(self) -> Callable[[str], str | None]:
        group_sets = [(''.join(group), set(group)) for group in self.groups]

        def get_group(name: str) -> str | None:
            if self.suspected_case:
                name = name.removesuffix('?')
            name_set = set(name.split('/'))
            return next((g for g, s in group_sets if name_set <= s), None)

        return get_group

    def get_group_index_mapping(self) -> dict[str, int]:
        return {''.join(group): i for i, group in enumerate(self.groups)}

    @property
    def group_name_str(self) -> str:
        return ', '.join(''.join(group) for group in self.groups)

    @property
    def ref_name_str(self) -> str:
        if self.ref_names is not None:
            ref_names = ', '.join(self.ref_names)
            if self.true_ref:
                description = f'ref {ref_names}, keep {self.keep}'
            else:
                description = f'of {ref_names}, keep {self.keep}'
        elif self.keep != 'all':
            description = f'keep {self.keep}'
        else:
            description = 'original'
        if self.label_name is not None:
            description = f'{description}, on {self.label_name}'
        return description

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
