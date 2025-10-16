from time import perf_counter_ns
from typing import override

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.models import RawModel
from iatreion.rrl import get_raw_samples

from .base import Trainer, TrainerReturn


class RawModelTrainer(Trainer):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        train_config: TrainConfig,
        model: RawModel,
    ) -> None:
        super().__init__(dataset_config, train_config)
        self.model = model
        self.samples = get_raw_samples(dataset_config, train_config)
        self.group_mapping = train_config.get_group_index_mapping()

    @override
    def train_step(self) -> TrainerReturn:
        X_train, y_train, X_test, y_test = next(self.samples)
        y_true = y_test.map(self.group_mapping).to_numpy()
        start = perf_counter_ns()
        self.model.fit(X_train, y_train)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity = self.model.predict(X_test, y_test)
        return training_time, y_true, y_score, complexity

    @override
    def train_final(self) -> None:
        X_train, y_train, _, _ = next(self.samples)
        self.model.fit(X_train, y_train)
