from time import perf_counter_ns
from typing import Any, override

from iatreion.configs import RrlConfig
from iatreion.rrl import get_samples
from iatreion.rrl.experiment import test_model, train_model
from iatreion.utils import set_seed_torch

from .base import Trainer, TrainerReturn


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(
            config.dataset, config.train, epilog_callback=config.epilog_callback
        )
        self.config = config
        self.samples = get_samples(config.dataset, config.train)
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
    def train_step(self) -> TrainerReturn:
        # HACK: Reset the seed for each training step to ensure reproducibility
        set_seed_torch(self.train_config.seed)
        samples = next(self.samples)
        start = perf_counter_ns()
        train_model(self.config, self.save_model_callback, samples)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity = test_model(self.config, self.save_model_callback, samples)
        return training_time, samples[-2], y_score, samples[-1], {'Log#E': complexity}

    @override
    def train_final(self) -> None:
        # HACK: Ditto albeit with only one training step
        set_seed_torch(self.train_config.seed)
        samples = next(self.samples)
        train_model(self.config, self.save_model_callback, samples)
