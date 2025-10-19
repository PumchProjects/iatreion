from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from .dataset import DatasetConfig
from .train import TrainConfig
from .utils import get_best_exp_root, get_rrl_file, register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class RrlConfig:
    dataset: DatasetConfig

    train: TrainConfig

    epoch: Annotated[int, Parameter(name=['--epoch', '-e'])] = 41
    'Set the total epoch.'

    batch_size: Annotated[int, Parameter(name=['--batch_size', '-bs'])] = 64
    'Set the batch size.'

    learning_rate: Annotated[float, Parameter(name=['--learning_rate', '-lr'])] = 0.01
    'Set the initial learning rate.'

    lr_decay_rate: Annotated[float, Parameter(name=['--lr_decay_rate', '-lrdr'])] = 0.75
    'Set the learning rate decay rate.'

    lr_decay_epoch: Annotated[int, Parameter(name=['--lr_decay_epoch', '-lrde'])] = 10
    'Set the learning rate decay epoch.'

    weight_decay: Annotated[float, Parameter(name=['--weight_decay', '-wd'])] = 0.0
    'Set the weight decay (L2 penalty).'

    log_iter: Annotated[int, Parameter(name=['--log_iter', '-li'])] = 500
    'The number of iterations (batches) to log once.'

    save_interval: Annotated[int, Parameter(name=['--save_interval', '-si'])] = 100
    'The number of epochs to save the model based on training loss (when save_best=False), or the number of iterations (batches) to save the model based on validation F1 (when save_best=True).'

    nlaf: Annotated[bool, Parameter(negative='')] = False
    'Use novel logical activation functions to take less time and GPU memory usage. We recommend trying (alpha, beta, gamma) in {(0.999, 8, 1), (0.999, 8, 3), (0.9, 3, 3)}'

    alpha: float = 0.999
    'Set the alpha for NLAF.'

    beta: int = 8
    'Set the beta for NLAF.'

    gamma: int = 1
    'Set the gamma for NLAF.'

    temp: float = 1.0
    'Set the temperature.'

    use_not: Annotated[bool, Parameter(negative='')] = False
    'Use the NOT (~) operator in logical rules. It will enhance model capability but make the RRL more complex.'

    save_best: Annotated[bool, Parameter(negative='')] = False
    'Save the model with best performance on the validation set.'

    skip: Annotated[bool, Parameter(negative='')] = False
    'Use skip connections when the number of logical layers is greater than 2.'

    estimated_grad: Annotated[bool, Parameter(negative='')] = False
    'Use estimated gradient.'

    weighted: Annotated[bool, Parameter(negative='')] = False
    'Use weighted loss for imbalanced data.'

    print_rule: Annotated[bool, Parameter(negative='')] = False
    'Print the rules.'

    structure: Annotated[str, Parameter(name=['--structure', '-s'])] = '5@64'
    'Set the number of nodes in the binarization layer and logical layers. E.g., 10@64, 10@64@32@16.'

    def __post_init__(self) -> None:
        self.dataset.simple = False
        over_sampler = str(self.train.over_sampler).upper()
        folder_name = (
            f'e{self.epoch}_os{over_sampler}_mns{self.train.min_n_samples}_bs{self.batch_size}'
            f'_lr{self.learning_rate}_lrdr{self.lr_decay_rate}_lrde{self.lr_decay_epoch}_wd{self.weight_decay}'
            f'_si{self.save_interval}_useNOT{self.use_not}_saveBest{self.save_best}_useSkip{self.skip}'
            f'_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}_temp{self.temp}_L{self.structure}'
        )
        register_log_dir(self.dataset, self.train, 'rrl', folder_name)

    def get_best_exp_root(self) -> tuple[Path, float | None]:
        return get_best_exp_root(self.dataset.name_str, self.train, final=False)

    @property
    def folder_path(self) -> str:
        return str(self.train.log_dir)

    @property
    def rrl_file(self) -> str:
        return str(get_rrl_file(self.train.log_dir, self.train))
