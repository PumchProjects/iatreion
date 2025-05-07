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

    @override
    def train_step(self) -> TrainerReturn:
        X_train, y_train, X_test, y_test = get_raw_samples(
            self.dataset_config, self.train_config
        )
        self.model.fit(X_train, y_train)
        y_score, complexity = self.model.predict(X_test, y_test)
        return y_test, y_score, complexity
