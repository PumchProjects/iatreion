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
        sub_recorders = {
            name: Recorder(self.train_config) for name in self.dataset_config.names
        }
        with progress:
            fold_task = progress.add_task('Fold:', total=self.train_config.n_folds)
            for fold in range(self.train_config.n_folds):
                self.train_config.ith_kfold = fold
                data_task = progress.add_task(
                    'Data:', total=len(self.dataset_config.names)
                )
                for name, sub_recorder in sub_recorders.items():
                    self.train_config.cur_name = name
                    results = self.train_step()
                    logger.info(sub_recorder.record(results))
                    progress.update(data_task, advance=1)
                logger.info(recorder.record_from(sub_recorders.values()))
                progress.update(fold_task, advance=1)
                progress.remove_task(data_task)
            progress.remove_task(fold_task)
        for name, sub_recorder in sub_recorders.items():
            with self.train_config.logging(name):
                logger.info(sub_recorder.finish())
        with self.train_config.logging('all'):
            logger.info(recorder.finish())
