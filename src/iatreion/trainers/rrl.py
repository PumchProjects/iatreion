from time import perf_counter_ns
from typing import Any, override

from iatreion.configs import RrlConfig
from iatreion.rrl import get_samples
from iatreion.rrl.experiment import test_model, train_model
from iatreion.utils import set_seed_torch

from .base import Trainer, TrainerReturn


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config.dataset, config.train)
        self.config = config
        self.samples = get_samples(config.dataset, config.train)
        self.model: Any = None
        self.weight: float | None = None
        set_seed_torch(self.train_config.seed)

    def save_model_callback(
        self, model: Any | None = None, weight: float | None = None
    ) -> tuple[Any, float]:
        if model is not None and weight is not None:
            self.model = model
            self.weight = weight
        assert self.weight is not None
        return self.model, self.weight

    @override
    def train_step(self) -> TrainerReturn:
        samples = next(self.samples)
        start = perf_counter_ns()
        train_model(self.config, self.save_model_callback, samples)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity, weight = test_model(
            self.config, self.save_model_callback, samples
        )
        return (
            training_time,
            samples[-2],
            y_score,
            samples[-1],
            {'Log#E': complexity, 'ValF1': (weight, '.2%')},
        )

    @override
    def train_final(self) -> None:
        samples = next(self.samples)
        train_model(self.config, self.save_model_callback, samples)
