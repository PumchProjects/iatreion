from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from .model_base import ModelConfig


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
    'The number of epochs to save the model based on training loss (when val_size=None), or the number of iterations (batches) to save the model based on validation F1 (when val_size is set).'

    nlaf: bool = False
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

    print_rule: bool = False
    'Print the rules.'

    structure: Annotated[str, Parameter(alias='-s')] = '5@64'
    'Set the number of nodes in the binarization layer and logical layers. E.g., 10@64, 10@64@32@16.'

    debug: Annotated[bool, Parameter(alias='-D')] = False
    'Whether to enable debug mode.'

    _folder_name: str | None = None

    @property
    def folder_path(self) -> Path:
        return self.train._log_dir / 'events'

    def __post_init__(self) -> None:
        if self.debug:
            over_sampler = str(self.train.over_sampler).upper()
            self._folder_name = (
                f'e{self.epoch}_os{over_sampler}_mns{self.train.min_n_samples}_bs{self.batch_size}'
                f'_lr{self.learning_rate}_lrdr{self.lr_decay_rate}_lrde{self.lr_decay_epoch}_wd{self.weight_decay}'
                f'_si{self.save_interval}_useNOT{self.use_not}_valSize{self.train.val_size}_useSkip{self.skip}'
                f'_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}_temp{self.temp}_L{self.structure}'
            )
        self.register_log_dir('rrl', folder_name=self._folder_name)
        self.folder_path.mkdir(parents=True, exist_ok=True)
