from typing import override

from iatreion.rrl import RrlConfig, test_model, train_model

from .base import Trainer


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__()
        self.config = config

    @override
    def train_step(self, fold: int) -> None:
        self.config.ith_kfold = fold
        train_model(self.config)
        test_model(self.config)
