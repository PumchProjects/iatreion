import os
from typing import override

from iatreion.configs import RrlConfig
from iatreion.models import ModelReturn
from iatreion.rrl import Samples
from iatreion.rrl.experiment import test_model, train_model
from iatreion.utils import set_seed_torch

from .base import Trainer


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config.dataset, config.train)
        self.config = config
        set_seed_torch(self.train_config.seed)

    @override
    def train_step(self, samples: Samples) -> ModelReturn:
        train_model(self.config, samples)
        y_score, complexity = test_model(self.config, samples)
        os.remove(self.config.model)
        return y_score, complexity
