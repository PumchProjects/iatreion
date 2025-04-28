from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory


@Parameter(name='*')
@dataclass
class RrlConfig:
    data_prefix: Annotated[ExistingDirectory, Parameter(name=['--prefix', '-p'])]
    'Prefix of the data files'

    data_set: Annotated[str, Parameter(name=['--data_set', '-d'])] = 'tic-tac-toe'
    'Set the data set for training. All the data sets in the dataset folder are available.'

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

    ith_kfold: Annotated[int, Parameter(name=['--ith_kfold', '-ki'])] = 0
    'Do the i-th 5-fold validation, 0 <= ki < 5.'

    log_iter: Annotated[int, Parameter(name=['--log_iter', '-li'])] = 500
    'The number of iterations (batches) to log once.'

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

    # TODO: why cannot use field(init=False) here?
    _folder_path: Annotated[Path, Parameter(parse=False)] = field(default_factory=Path)

    def __post_init__(self) -> None:
        folder_name = (
            f'e{self.epoch}_bs{self.batch_size}_lr{self.learning_rate}_lrdr{self.lr_decay_rate}'
            f'_lrde{self.lr_decay_epoch}_wd{self.weight_decay}_useNOT{self.use_not}_saveBest{self.save_best}'
            f'_useSkip{self.skip}_alpha{self.alpha}_beta{self.beta}_gamma{self.gamma}_temp{self.temp}_L{self.structure}'
        )
        self._folder_path = Path('log_folder') / self.data_set / folder_name
        self._folder_path.mkdir(parents=True, exist_ok=True)

    @property
    def folder_path(self) -> str:
        return str(self._folder_path)

    @property
    def model(self) -> str:
        return str(self._folder_path / f'model_{self.ith_kfold}.pth')

    @property
    def rrl_file(self) -> str:
        return str(self._folder_path / f'rrl_{self.ith_kfold}.txt')

    @property
    def plot_file(self) -> str:
        return str(self._folder_path / f'plot_file_{self.ith_kfold}.pdf')

    @property
    def log(self) -> str:
        return str(self._folder_path / f'log_{self.ith_kfold}.txt')

    @property
    def test_res(self) -> str:
        return str(self._folder_path / f'test_res_{self.ith_kfold}.txt')
