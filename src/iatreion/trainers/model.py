from typing import override

from iatreion.models import Model
from iatreion.train_utils import TrainStepContext
from iatreion.utils import Timer

from .base import Trainer, TrainerReturn


class ModelTrainer(Trainer):
    def __init__(self, model: Model) -> None:
        super().__init__(model.config)
        self.model = model

    @override
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn:
        # HACK: Validation set is not used for other models
        with Timer() as timer:
            self.model.fit(ctx)
        y_score, complexity = self.model.predict(ctx)
        return timer.duration, ctx.test_data[1], y_score, complexity

    @override
    def train_final(self, ctx: TrainStepContext) -> None:
        # HACK: Validation set is not used for other models
        self.model.fit(ctx)
