from time import perf_counter_ns
from typing import override

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.models import Model
from iatreion.rrl import get_samples

from .base import Trainer, TrainerReturn


class ModelTrainer(Trainer):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        train_config: TrainConfig,
        model: Model,
    ) -> None:
        super().__init__(dataset_config, train_config)
        self.model = model

    @override
    def train_step(self) -> TrainerReturn:
        _, X_train, y_train, X_test, y_test = get_samples(
            self.dataset_config, self.train_config
        )
        start = perf_counter_ns()
        self.model.fit(X_train, y_train)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity = self.model.predict(X_test, y_test)
        return training_time, y_test, y_score, complexity
