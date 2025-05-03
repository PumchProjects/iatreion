from typing import override

from rich.progress import Progress

from iatreion.configs import DatasetConfig, ModelConfig, TrainConfig
from iatreion.models import Model
from iatreion.rrl import get_samples

from .base import Trainer


class ModelTrainer(Trainer):
    def __init__(
        self,
        dataset: DatasetConfig,
        config: ModelConfig,
        train: TrainConfig,
        model: Model,
    ) -> None:
        super().__init__(train)
        self.dataset_config = dataset
        self.model_config = config
        self.train_config = train
        self.model = model

    @override
    def train_step(self, fold: int, progress: Progress) -> None:
        self.model_config.ith_kfold = fold
        _, X_train, y_train, X_test, y_test = get_samples(
            self.dataset_config,
            self.model_config,
            self.train_config,
        )
        self.model.fit(X_train, y_train)
        y_score, complexity = self.model.predict(X_test, y_test)
