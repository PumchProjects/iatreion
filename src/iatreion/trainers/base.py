from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.utils import logger, progress

from .recorder import Recorder

type TrainerReturn = tuple[float, ArrayLike, ArrayLike, dict[str, float]]


class Trainer(ABC):
    def __init__(
        self, dataset_config: DatasetConfig, train_config: TrainConfig
    ) -> None:
        self.dataset_config = dataset_config
        self.train_config = train_config

    @abstractmethod
    def train_step(self) -> TrainerReturn: ...

    def train(self) -> None:
        recorder = Recorder(self.train_config)
        with progress:
            fold_task = progress.add_task('Fold:', total=self.train_config.n_splits)
            for fold in range(self.train_config.n_splits):
                logger.info(
                    f'[bold green]Fold[/] {fold + 1}/{self.train_config.n_splits}',
                    extra={'markup': True},
                )
                self.train_config.ith_kfold = fold
                training_time, y_true, y_score, complexity = self.train_step()
                recorder.record(training_time, y_true, y_score, complexity)
                progress.update(fold_task, advance=1)
                logger.info('')
        recorder.finish()
