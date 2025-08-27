from abc import ABC, abstractmethod

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.utils import logger, progress

from .recorder import Recorder, TrainerReturn


class Trainer(ABC):
    def __init__(
        self, dataset_config: DatasetConfig, train_config: TrainConfig
    ) -> None:
        self.dataset_config = dataset_config
        self.train_config = train_config

    @abstractmethod
    def train_step(self) -> TrainerReturn: ...

    @abstractmethod
    def train_final(self) -> None: ...

    def train(self) -> None:
        if self.train_config.final:
            self.train_final()
            return
        recorder = Recorder(self.train_config)
        with progress:
            fold_task = progress.add_task('Fold:', total=self.train_config.n_folds)
            for fold in range(self.train_config.n_folds):
                logger.info(
                    f'[bold green]Fold[/] {fold + 1}/{self.train_config.n_folds}',
                    extra={'markup': True},
                )
                self.train_config.ith_kfold = fold
                results = self.train_step()
                recorder.record(results)
                progress.update(fold_task, advance=1)
                logger.info('')
        recorder.finish()
