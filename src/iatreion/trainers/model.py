from typing import override

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.rrl import TrainStepContext
from iatreion.utils import Timer

from .base import Trainer, TrainerReturn


class ModelTrainer(Trainer):
    def __init__(self, config: ModelConfig, model: Model) -> None:
        super().__init__(config)
        self.model = model

    @override
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn:
        # HACK: Validation set is not used for other models
        with Timer() as timer:
            self.model.fit(*ctx.train_data)
        y_score, complexity = self.model.predict(ctx, *ctx.test_data)
        return timer.duration, ctx.test_data[1], y_score, complexity

    @override
    def train_final(self, ctx: TrainStepContext) -> None:
        # HACK: Validation set is not used for other models
        self.model.fit(*ctx.train_data)
