from time import perf_counter_ns
from typing import override

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.models import Model
from iatreion.rrl import TrainStepContext

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
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn:
        # HACK: Validation set is not used for other models
        start = perf_counter_ns()
        self.model.fit(*ctx.train_data)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity = self.model.predict(*ctx.test_data)
        return training_time, ctx.test_data[1], y_score, complexity

    @override
    def train_final(self) -> None:
        # HACK: Validation set is not used for other models
        _, X_train, y_train, *_ = next(self.samples)
        self.model.fit(X_train, y_train)
