from typing import override

from rich.progress import Progress

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.models import Model
from iatreion.rrl import get_samples

from .base import Trainer


class ModelTrainer(Trainer):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        train_config: TrainConfig,
        model: Model,
    ) -> None:
        super().__init__(train_config)
        self.dataset_config = dataset_config
        self.model = model

    @override
    def train_step(self, fold: int, progress: Progress) -> None:
        self.train_config.ith_kfold = fold
        _, X_train, y_train, X_test, y_test = get_samples(
            self.dataset_config,
            self.train_config,
        )
        self.model.fit(X_train, y_train)
        y_score, complexity = self.model.predict(X_test, y_test)
