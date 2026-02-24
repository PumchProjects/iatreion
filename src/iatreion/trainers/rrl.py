from typing import Any, override

from iatreion.configs import RrlConfig
from iatreion.rrl import TrainStepContext
from iatreion.rrl.experiment import test_model, train_model
from iatreion.utils import Timer, set_seed_torch

from .base import Trainer, TrainerReturn


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model: Any = None
        self.state_dict: dict[str, Any] = {}
        self.metrics: tuple[float, ...] = ()

    def save_model_callback(
        self, *args: Any
    ) -> tuple[Any, dict[str, Any], tuple[float, ...]]:
        if len(args) > 0:
            self.model, self.state_dict, self.metrics = args
        return self.model, self.state_dict, self.metrics

    @override
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn:
        # HACK: Reset the seed for each training step to ensure reproducibility
        set_seed_torch(self.train_config.seed)
        with Timer() as timer:
            train_model(self.config, self.save_model_callback, ctx)
        y_score, complexity = test_model(self.config, self.save_model_callback, ctx)
        return timer.duration, ctx.test_data[1], y_score, {'Log#E': complexity}

    @override
    def train_final(self, ctx: TrainStepContext) -> None:
        # HACK: Ditto albeit with only one training step
        set_seed_torch(self.train_config.seed)
        train_model(self.config, self.save_model_callback, ctx)
