from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import (
    Directory,
    ExistingDirectory,
    ExistingFile,
    NonNegativeInt,
    PositiveInt,
)
from cyclopts.validators import Number

from iatreion.utils import (
    add_file_handler,
    expand_range,
    remove_file_handler,
    set_device,
    set_seed,
)

from .feature_selection import FeatureSelectionConfig

type UnderSamplerName = Literal['random']
type AggregationMethod = Literal[
    'average', 'concat', 'calibrated-concat', 'calibrated-fusion'
]
type MissingValueStrategy = Literal['simple', 'limix', 'none']
type DiscreteProcessingStrategy = Literal['onehot', 'minmax', 'none']

INNER_SPLIT_AGGREGATES: tuple[AggregationMethod, ...] = (
    'calibrated-concat',
    'calibrated-fusion',
)


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
'calibrated-concat': concatenate features into one RRL, calibrate its logit,
and tune operating thresholds.
'calibrated-fusion': train one RRL per modality, calibrate each modality logit,
and combine available modalities with equal-weight late fusion.
"""

    preprocess: Annotated[bool, Parameter(negative='--no-pp')] = True
    'Whether to preprocess the data (e.g., filling missing values, normalization).'

    missing_value_strategy: MissingValueStrategy = 'simple'
    """Missing-value handling strategy.
'simple': use mode for unordered, median for ordered/discrete, mean for continuous.
'limix': use the LimiX reconstruction model and only fill missing entries.
'none': keep missing values unchanged.
"""

    normalize_continuous: Annotated[
        bool, Parameter(negative='--no-normalize-continuous')
    ] = True
    'Whether to z-score normalize continuous features.'

    discrete_processing: DiscreteProcessingStrategy = 'onehot'
    """Processing strategy for non-continuous features after optional under-sampling.
'onehot': one-hot encode categorical features.
'minmax': min-max scale categorical codes to [0, 1].
'none': keep categorical codes unchanged.
"""

    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    'Supervised feature-selection settings.'

    eval_names: Annotated[list[str], Parameter(consume_multiple=True)] = field(
        default_factory=list
    )
    'Optional subset of modalities to evaluate/fuse while keeping splits based on all input modalities.'

    n_outer_splits: int = 5
    'Number of splits for outer cross-validation.'

    n_inner_splits: int = 5
    "Number of splits for inner cross-validation, used when aggregate='calibrated-fusion' or 'calibrated-concat'."

    use_clinical_threshold: Annotated[
        bool, Parameter(negative='--no-clinical-threshold')
    ] = True
    'Whether to calculate and record the clinical recall threshold.'

    clinical_threshold_label: str = ''
    'Class label whose recall is targeted by the clinical threshold.'

    clinical_threshold_recall: Annotated[
        float, Parameter(validator=Number(gt=0, lt=1))
    ] = 0.9
    'Target recall for the clinical threshold.'

    device_id: Annotated[str, Parameter(alias='-i')] = '0'
    "CUDA device IDs for training, e.g. '0', '0,1', or '0-7'. Default is 0."

    final: Annotated[bool, Parameter(alias='-f')] = False
    'Whether to use the whole dataset for training or testing.'

    under_sampler: UnderSamplerName | None = None
    'Under-sampling method to use.'

    target_n_samples: NonNegativeInt = 0
    'Maximum number of samples to keep for each class after under-sampling. Use 0 to balance to the smallest class.'

    limix_python_path: ExistingFile | None = None
    'Python interpreter used for LimiX-based missing-value imputation.'

    limix_repo_path: ExistingDirectory | None = None
    'Path to the LimiX repository used for missing-value imputation.'

    limix_model_path: ExistingFile | None = None
    'Path to the pre-trained LimiX model file used for missing-value imputation.'

    limix_inference_config_path: ExistingFile | None = None
    'Optional override for the LimiX missing-value inference config file.'

    limix_device: str = 'cuda'
    'Device passed to the LimiX missing-value worker.'

    val_size: float | int | None = None
    """If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
If int, represents the absolute number of validation samples.
If None (default), no validation set is used.
For discrete RRL, validation set is used for optimization when val_size is set.
"""

    suspected_case: bool = False
    'Whether to include suspected cases in training.'

    label_name: str | None = None
    'Label column name in the data files. If not set, determined automatically.'

    seed: int = 42
    'Random seed for reproducibility.'

    plot_roc: Annotated[bool, Parameter(negative=None)] = True
    'Plot ROC curve.'

    bootstrap_samples: PositiveInt = 1000
    'Number of bootstrap resamples used to estimate confidence intervals.'

    ci_level: Annotated[float, Parameter(validator=Number(gt=0, lt=1))] = 0.95
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
    def clinical_threshold_index(self) -> int:
        return self.get_group_index_mapping()[self.clinical_threshold_label]

    @property
    def group_name_str(self) -> str:
        return '_'.join(''.join(group) for group in self._groups)

    @property
    def ref_name_str(self) -> str:
        # HACK: Don't include `preprocess` here since RRL needs preprocessed data while discrete RRL doesn't
        descriptions = [self.aggregate]
        if self.label_name is not None:
            descriptions.append(f'on {self.label_name}')
        return '_'.join(descriptions)

    @property
    def eval_name_str(self) -> str:
        return '_'.join(self.eval_names)

    @property
    def n_outer_folds(self) -> int:
        return self.n_outer_splits

    @property
    def n_inner_folds(self) -> int:
        return self.n_inner_splits

    @property
    def n_folds(self) -> int:
        # HACK: Coupled with get_train_iterator()
        if self.final:
            return 1
        if self.aggregate in INNER_SPLIT_AGGREGATES:
            return self.n_outer_folds * (self.n_inner_folds + 1)
        return self.n_outer_folds

    @property
    def device_ids(self) -> list[int]:
        devices: list[int] = []
        for item in self.device_id.split(','):
            item = item.strip()
            if not item:
                continue
            if '-' not in item:
                devices.append(int(item))
                continue
            start, end = (int(edge) for edge in item.split('-', maxsplit=1))
            devices.extend(range(start, end + 1))
        return devices or [0]

    @property
    def num_class(self) -> int:
        return len(self._groups)

    def __post_init__(self) -> None:
        set_device(self.device_ids)
        set_seed(self.seed)
        self.set_groups()
        if self.num_class > 2:
            # HACK: Disable ROC plot for multiclass classification
            self.plot_roc = False
        if self.use_clinical_threshold:
            if not self.clinical_threshold_label:
                self.clinical_threshold_label = self._sorted_group_names[0]
            if self.clinical_threshold_label not in self.get_group_index_mapping():
                raise ValueError(
                    'clinical_threshold_label must be one of '
                    f'{", ".join(self._sorted_group_names)}.'
                )
        self.validate_feature_selection()
        self.validate_preprocessing()

    def validate_feature_selection(self) -> None:
        self.feature_selection.validate()

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
