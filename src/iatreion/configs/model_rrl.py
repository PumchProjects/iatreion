from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import PositiveFloat
from cyclopts.validators import Number

from .model_base import ModelConfig

type MissingAwareMode = Literal['original', 'improved']
type RrlValidationMetric = Literal['f1', 'auroc', 'auprc']


@Parameter(name='*')
@dataclass(kw_only=True)
class RrlConfig(ModelConfig):
    epoch: Annotated[int, Parameter(alias='-e')] = 41
    'Set the total epoch.'

    batch_size: Annotated[int, Parameter(alias='-bs')] = 64
    'Set the batch size.'

    learning_rate: Annotated[float, Parameter(alias='-lr')] = 0.01
    'Set the initial learning rate.'

    lr_decay_rate: Annotated[float, Parameter(alias='-lrdr')] = 0.75
    'Set the learning rate decay rate.'

    lr_decay_epoch: Annotated[int, Parameter(alias='-lrde')] = 10
    'Set the learning rate decay epoch.'

    weight_decay: Annotated[float, Parameter(alias='-wd')] = 0.0
    'Set the weight decay (L2 penalty).'

    log_iter: Annotated[int, Parameter(alias='-li')] = 500
    'The number of iterations (batches) to log once.'

    save_interval: Annotated[int, Parameter(alias='-si')] = 100
    'The number of epochs to save the model based on training loss (when val_size=None), or the number of iterations (batches) to save the model based on the validation metric (when val_size is set).'

    validation_metric: Annotated[RrlValidationMetric, Parameter(alias='-vm')] = 'f1'
    'Validation metric used for checkpoint selection and early stopping when val_size is set.'

    early_stop_patience: Annotated[int | None, Parameter(alias='-esp')] = None
    'Number of validation checks with no sufficient validation-metric improvement before early stopping. Disabled when None or <= 0.'

    early_stop_min_delta: Annotated[float, Parameter(alias='-esd')] = 0.0
    'Minimum required increase in the validation metric to reset early-stopping patience.'

    label_smoothing: Annotated[float, Parameter(alias='-ls')] = 0.0
    'Label smoothing factor for cross-entropy loss.'

    max_grad_norm: Annotated[float | None, Parameter(alias='-mgn')] = 5.0
    'Max gradient norm for clipping. Disabled when None or <= 0.'

    nlaf: bool = True
    'Use novel logical activation functions to take less time and GPU memory usage. We recommend trying (alpha, beta, gamma) in {(0.999, 8, 1), (0.999, 8, 3), (0.9, 3, 3)}'

    alpha: float = 0.999
    'Set the alpha for NLAF.'

    beta: int = 8
    'Set the beta for NLAF.'

    gamma: int = 1
    'Set the gamma for NLAF.'

    temp: float = 1.0
    'Set the temperature.'

    use_not: bool = False
    'Use the NOT (~) operator in logical rules. It will enhance model capability but make the RRL more complex.'

    skip: bool = False
    'Use skip connections when the number of logical layers is greater than 2.'

    estimated_grad: bool = False
    'Use estimated gradient.'

    weighted: bool = False
    'Use weighted loss for imbalanced data.'

    conjunction_only: bool = False
    'Use only conjunction logical layers. When false, each logical layer includes both conjunction and disjunction branches.'

    structure: Annotated[str, Parameter(alias='-s')] = '5@64'
    'Set the number of nodes in the binarization layer and logical layers. E.g., 10@64, 10@64@32@16.'

    debug: Annotated[bool, Parameter(alias='-D')] = False
    'Whether to enable debug mode.'

    missing_aware_mode: Annotated[MissingAwareMode, Parameter(alias='-v')] = 'original'
    'Select `original` RRL behavior or the improved missing-aware RRL with mask-aware + coverage-gated logic.'

    coverage_tau: Annotated[
        float, Parameter(name='--tau', validator=Number(gte=0, lte=1))
    ] = 0.5
    'Coverage threshold used by the improved RRL gate.'

    coverage_kappa: Annotated[PositiveFloat, Parameter(name='--kappa')] = 0.1
    'Soft gate sharpness used by the improved RRL gate.'

    _folder_name: str | None = None

    @property
    def folder_path(self) -> Path:
        return self.train._log_dir / 'events'

    def __post_init__(self) -> None:
        self.train._encode = True
        if self.missing_aware_mode == 'improved':
            self.train.missing_value_strategy = 'none'
            self.train.validate_preprocessing()
        if self.debug:
            under_sampler = str(self.train.under_sampler).upper()
            self._folder_name = (
                f'e{self.epoch}_us{under_sampler}_tns{self.train.target_n_samples}_bs{self.batch_size}'
                f'_lr{self.learning_rate}_lrdr{self.lr_decay_rate}_lrde{self.lr_decay_epoch}_wd{self.weight_decay}'
                f'_si{self.save_interval}_useNOT{self.use_not}_valSize{self.train.val_size}_useSkip{self.skip}'
                f'_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}_temp{self.temp}'
                f'_conjOnly{self.conjunction_only}_L{self.structure}'
                f'_missingMode{self.missing_aware_mode}_tau{self.coverage_tau}_kappa{self.coverage_kappa}'
                f'_vm{self.validation_metric}_esp{self.early_stop_patience}_esd{self.early_stop_min_delta}'
                f'_ls{self.label_smoothing}_mgn{self.max_grad_norm}'
            )
        self.register_log_dir('rrl', folder_name=self._folder_name)
