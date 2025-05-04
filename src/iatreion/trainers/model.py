from typing import override

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.models import Model, ModelReturn
from iatreion.rrl import Samples

from .base import Trainer


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
    def train_step(self, samples: Samples) -> ModelReturn:
        _, X_train, y_train, X_test, y_test = samples
        self.model.fit(X_train, y_train)
        y_score, complexity = self.model.predict(X_test, y_test)
        return y_score, complexity
