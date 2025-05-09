from dataclasses import dataclass

from cyclopts import Parameter

from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class GosdtConfig:
    dataset: DatasetConfig

    train: TrainConfig

    guess_th: bool = True
    'Whether to guess thresholds for the dataset.'

    guess_lb: bool = True
    'Whether to guess lower bounds for the dataset.'

    gbdt_n_est: int = 40
    'The number of estimators for the Gradient Boosting Classifier.'

    gbdt_max_depth: int = 1
    'The maximum depth of the trees for the Gradient Boosting Classifier.'

    regularization: float = 0.001
    'The regularization penalty incurred for each leaf in the model.'

    similar_support: bool = False
    'A boolean flag enabling the similar support bound implemented via a distance index.'

    depth_budget: int = 6
    'Sets the maximum tree depth for a solution model.'

    time_limit: int = 60
    'A time limit (in seconds) upon which the algorithm will terminate.'

    worker_limit: int = 20
    'Set the number of parallel processes to use for training and prediction.'

    verbose: bool = True
    'A boolean flag enabling verbose output.'

    def __post_init__(self) -> None:
        self.train.log_dir = (
            self.train.log_root / self.dataset.name / self.train.groups / 'gosdt'
        )
        add_file_handler(self.train.log_dir / 'log.txt')
        self.train.record_auc = False
