import os
from time import perf_counter_ns
from typing import override

from iatreion.configs import RrlConfig
from iatreion.rrl import get_samples
from iatreion.rrl.experiment import test_model, train_model
from iatreion.utils import set_seed_torch

from .base import Trainer, TrainerReturn


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config.dataset, config.train)
        self.config = config
        set_seed_torch(self.train_config.seed)

    @override
    def train_step(self) -> TrainerReturn:
        samples = get_samples(self.dataset_config, self.train_config)
        start = perf_counter_ns()
        train_model(self.config, samples)
        end = perf_counter_ns()
        training_time = (end - start) / 1e9
        y_score, complexity = test_model(self.config, samples)
        os.remove(self.config.model)
        return training_time, samples[4], y_score, {'Log#E': complexity}

    @override
    def train_final(self) -> None:
        samples = get_samples(self.dataset_config, self.train_config)
        train_model(self.config, samples)
        os.remove(self.config.model)
