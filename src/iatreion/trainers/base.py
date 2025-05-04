from abc import ABC, abstractmethod

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.models import ModelReturn
from iatreion.rrl import Samples, get_samples
from iatreion.utils import progress

from .recorder import Recorder


class Trainer(ABC):
    def __init__(
        self, dataset_config: DatasetConfig, train_config: TrainConfig
    ) -> None:
        self.dataset_config = dataset_config
        self.train_config = train_config

    @abstractmethod
    def train_step(self, samples: Samples) -> ModelReturn: ...

    def train(self) -> None:
        recorder = Recorder(self.train_config)
        with progress:
            fold_task = progress.add_task('Fold:', total=self.train_config.n_splits)
            for fold in range(self.train_config.n_splits):
                self.train_config.ith_kfold = fold
                samples = get_samples(self.dataset_config, self.train_config)
                y_score, complexity = self.train_step(samples)
                recorder.record(samples[4], y_score, complexity)
                progress.update(fold_task, advance=1)
        recorder.finish()
