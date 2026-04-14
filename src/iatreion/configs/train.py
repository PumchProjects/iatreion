from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory, ExistingDirectory, ExistingFile, PositiveInt
from cyclopts.validators import Number

from iatreion.utils import (
    add_file_handler,
    expand_range,
    remove_file_handler,
    set_device,
    set_seed,
)

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
type AggregationMethod = Literal['average', 'concat', 'concats', 'stack']
type MissingValueStrategy = Literal['simple', 'limix', 'none']
type DiscreteProcessingStrategy = Literal['onehot', 'minmax', 'none']


@Parameter(name='*')
@dataclass(kw_only=True)
class TrainConfig:
    group_names: Annotated[
        list[str], Parameter(name=['--groups', '-g'], consume_multiple=True)
    ]
    'Group names of the data.'

    keep: Annotated[Literal['first', 'last'], Parameter(alias='-k')] = 'last'
    """Deduplication strategy for duplicated samples.
'first': keep the first sample of each patient.
'last': keep the last sample of each patient.
"""

    aggregate: Annotated[AggregationMethod, Parameter(alias='-a')] = 'average'
    """Aggregation strategy for multimodal samples of the same patient.
'average': simple average predictions of different modalities.
'concat': concatenate features of different modalities.
'concats': concatenate features of different modalities and adjust classification threshold.
'stack': late fusion by stacking predictions of different modalities as features for a meta-classifier.
"""

    preprocess: Annotated[bool, Parameter(negative='--no-pp')] = True
    'Whether to preprocess the data (e.g., filling missing values, normalization).'

    missing_value_strategy: Annotated[MissingValueStrategy, Parameter(alias='-mvs')] = (
        'simple'
    )
    """Missing-value handling strategy.
'simple': use mode for unordered, median for ordered/discrete, mean for continuous.
'limix': use the LimiX reconstruction model and only fill missing entries.
'none': keep missing values unchanged.
"""

    normalize_continuous: Annotated[
        bool,
        Parameter(name='--normalize-continuous', negative='--no-normalize-continuous'),
    ] = True
    'Whether to z-score normalize continuous features.'

    discrete_processing: Annotated[
        DiscreteProcessingStrategy, Parameter(alias='-dp')
    ] = 'onehot'
    """Processing strategy for non-continuous features after resampling.
'onehot': one-hot encode categorical features.
'minmax': min-max scale categorical codes to [0, 1].
'none': keep categorical codes unchanged.
"""

    true_ref: Annotated[bool, Parameter(alias='-tr')] = False
    'Align not only the test data, but also the training data to the reference data.'

    n_outer_splits: Annotated[int, Parameter(alias='-nos')] = 5
    'Number of splits for outer cross-validation.'

    n_outer_repeats: Annotated[int, Parameter(alias='-nor')] = 1
    'Number of repeats for outer cross-validation.'

    n_inner_splits: Annotated[int, Parameter(alias='-nis')] = 5
    "Number of splits for inner cross-validation, used when aggregate='stack'."

    n_inner_repeats: Annotated[int, Parameter(alias='-nir')] = 1
    "Number of repeats for inner cross-validation, used when aggregate='stack'."

    device_id: Annotated[int, Parameter(alias='-i')] = 0
    'Device ID for training. Default is 0.'

    final: Annotated[bool, Parameter(alias='-f')] = False
    'Whether to use the whole dataset for training or testing.'

    over_sampler: Annotated[SamplerName | None, Parameter(alias='-os')] = None
    'Over-sampling method to use.'

    min_n_samples: Annotated[int, Parameter(alias='-mns')] = 0
    'Minimum number of samples for each class after resampling.'

    limix_python_path: Annotated[ExistingFile | None, Parameter(alias='-lpp')] = None
    'Python interpreter used for LimiX-based missing-value imputation.'

    limix_repo_path: Annotated[ExistingDirectory | None, Parameter(alias='-lrp')] = None
    'Path to the LimiX repository used for missing-value imputation.'

    limix_model_path: Annotated[ExistingFile | None, Parameter(alias='-lmp')] = None
    'Path to the pre-trained LimiX model file used for missing-value imputation.'

    limix_inference_config_path: Annotated[
        ExistingFile | None, Parameter(alias='-lic')
    ] = None
    'Optional override for the LimiX missing-value inference config file.'

    limix_device: Annotated[str, Parameter(alias='-ld')] = 'cuda'
    'Device passed to the LimiX missing-value worker.'

    val_size: Annotated[float | int | None, Parameter(alias='-vs')] = None
    """If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
If int, represents the absolute number of validation samples.
If None (default), no validation set is used.
For discrete RRL, validation set is used for optimization when val_size is set.
"""

    suspected_case: Annotated[bool, Parameter(alias='-sc')] = False
    'Whether to include suspected cases in training.'

    label_name: Annotated[str | None, Parameter(alias='-ln')] = None
    'Label column name in the data files. If not set, determined automatically.'

    seed: int = 42
    'Random seed for reproducibility.'

    plot_roc: Annotated[bool, Parameter(negative=None)] = True
    'Plot ROC curve.'

    bootstrap_samples: Annotated[PositiveInt, Parameter(alias='-nbs')] = 1000
    'Number of bootstrap resamples used to estimate confidence intervals.'

    ci_level: Annotated[
        float, Parameter(validator=Number(gt=0, lt=1), alias='-cil')
    ] = 0.95
    'Confidence level in (0, 1) for bootstrap confidence intervals.'

    log_root: Directory = Path('logs')
    'Root directory for logs.'

    # TODO: why cannot use field(init=False) here?
    _log_dir: Directory = Path('logs')

    _base_pos: str = ''

    _label_pos: str = 'group_encrypted'

    _groups: list[list[str]] = field(default_factory=list[list[str]])

    _group_names: list[str] = field(default_factory=list[str])

    _sorted_group_names: list[str] = field(default_factory=list[str])

    _shuffle: bool = True

    _encode: bool = False

    def set_groups(self) -> None:
        if not self.group_names:
            raise ValueError('No valid groups found.')
        groups: list[list[str]] = []
        for group in self.group_names:
            group = expand_range(group)
            names: list[str] = []
            i = 0
            while i < len(group):
                if group[i].isupper() and i + 1 < len(group):
                    if group[i + 1] in ['<', '>']:
                        self._base_pos = 'AC 60'
                        names.append(group[i : i + 4])
                        i += 4
                    else:
                        self._base_pos = 'AC to 3'
                        names.append(group[i : i + 2])
                        i += 2
                else:
                    if group[i] in '12':
                        self._label_pos = 'group_Ab'
                    names.append(group[i])
                    i += 1
            groups.append(sorted(names))
        self._group_names = [''.join(group) for group in groups]
        self._groups = sorted(groups, key=lambda x: x[0])
        self._sorted_group_names = [''.join(group) for group in self._groups]
        if self.label_name is not None:
            self._base_pos = ''
            self._label_pos = self.label_name

    def get_name_group_mapping(self) -> Callable[[str], str | None]:
        group_sets = [(''.join(group), set(group)) for group in self._groups]

        def get_group(name: str) -> str | None:
            if self.suspected_case:
                name = name.removesuffix('?')
            name_set = set(name.split('/'))
            return next((g for g, s in group_sets if name_set <= s), None)

        return get_group

    def get_group_index_mapping(self) -> dict[str, int]:
        return {''.join(group): i for i, group in enumerate(self._groups)}

    @property
    def group_name_str(self) -> str:
        return ', '.join(''.join(group) for group in self._groups)

    @property
    def ref_name_str(self) -> str:
        # HACK: Don't include `preprocess` here since RRL needs preprocessed data while discrete RRL doesn't
        descriptions = [self.aggregate]
        if self.true_ref:
            descriptions.append('true-ref')
        if self.label_name is not None:
            descriptions.append(f'on {self.label_name}')
        return ', '.join(descriptions)

    @property
    def n_outer_folds(self) -> int:
        return self.n_outer_splits * self.n_outer_repeats

    @property
    def n_inner_folds(self) -> int:
        return self.n_inner_splits * self.n_inner_repeats

    @property
    def n_folds(self) -> int:
        # HACK: Coupled with get_train_iterator()
        if self.final:
            return 1
        if self.aggregate in ('concats', 'stack'):
            return self.n_outer_folds * (self.n_inner_folds + 1)
        return self.n_outer_folds

    @property
    def num_class(self) -> int:
        return len(self._groups)

    def __post_init__(self) -> None:
        set_device(self.device_id)
        set_seed(self.seed)
        self.set_groups()
        if self.num_class > 2:
            # HACK: Disable ROC plot for multiclass classification
            self.plot_roc = False
        self.validate_preprocessing()

    @property
    def resolved_limix_inference_config_path(self) -> Path:
        if self.limix_inference_config_path is not None:
            return self.limix_inference_config_path
        if self.limix_repo_path is None:
            raise ValueError('limix_repo_path must be set when using LimiX imputation.')
        return self.limix_repo_path / 'config' / 'reg_default_noretrieval_MVI.json'

    def validate_preprocessing(self) -> None:
        if not self.preprocess:
            return
        if self.over_sampler is not None and self.missing_value_strategy == 'none':
            raise ValueError(
                'Over-sampling requires missing_value_strategy to be "simple" or "limix".'
            )
        if self.missing_value_strategy != 'limix':
            return

        missing: list[str] = []
        if self.limix_python_path is None:
            missing.append('limix_python_path')
        if self.limix_repo_path is None:
            missing.append('limix_repo_path')
        if self.limix_model_path is None:
            missing.append('limix_model_path')
        if missing:
            joined = ', '.join(missing)
            raise ValueError(
                f'LimiX imputation requires the following parameters: {joined}.'
            )

        inference_config = self.resolved_limix_inference_config_path
        if not inference_config.is_file():
            raise ValueError(
                f'LimiX inference config file not found: {inference_config}'
            )

    @contextmanager
    def logging(self, name: str | Path) -> Generator[None, None, None]:
        filename = name if isinstance(name, Path) else self._log_dir / f'{name}.log'
        handler = add_file_handler(filename, format=False)
        try:
            yield
        finally:
            remove_file_handler(handler)

    def get_avg_log_file(self, name: str) -> Path:
        return self._log_dir / f'train_avg_{name}.log'

    def get_ci_log_file(self, name: str) -> Path:
        return self._log_dir / f'train_ci_{name}.log'

    def get_results_file(self, name: str) -> Path:
        return self._log_dir / f'results_{name}.npz'

    def get_roc_file(self, name: str) -> Path:
        return self._log_dir / f'roc_{name}.png'
