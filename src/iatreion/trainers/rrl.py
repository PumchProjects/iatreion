from time import perf_counter_ns
from typing import Any, override

from iatreion.configs import RrlConfig
from iatreion.exceptions import IatreionException
from iatreion.rrl import get_samples
from iatreion.rrl.experiment import test_model, train_model
from iatreion.utils import logger, set_seed_torch

from .base import Trainer, TrainerReturn


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config.dataset, config.train)
        self.config = config
        self.samples = get_samples(config.dataset, config.train)
        self.model: Any = None
        self.weight: float | None = None
        if config.train.final:
            try:
                _, self.weight = config.get_best_exp_root()
            except IatreionException:
                self.weight = 1.0
                logger.warning(
                    '[bold yellow]No previous best model found,'
                    f' the weight will be set to {self.weight}.'
                    ' Please set it manually later if needed.',
                    extra={'markup': True},
                )
        set_seed_torch(self.train_config.seed)

    def save_model_callback(self, model: Any | None) -> tuple[Any, float | None]:
        if model is not None:
            self.model = model
        return self.model, self.weight

    @override
    def train_step(self) -> TrainerReturn:
        samples = next(self.samples)
        start = perf_counter_ns()
        train_model(self.config, self.save_model_callback, samples)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity = test_model(self.config, self.model, samples)
        return training_time, samples[4], y_score, samples[5], {'Log#E': complexity}

    @override
    def train_final(self) -> None:
        samples = next(self.samples)
        train_model(self.config, self.save_model_callback, samples)
