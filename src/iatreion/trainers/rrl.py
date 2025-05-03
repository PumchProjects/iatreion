import os
from typing import override

from rich.progress import Progress

from iatreion.configs import RrlConfig
from iatreion.rrl import test_model, train_model

from .base import Trainer


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config.train)
        self.config = config

    @override
    def train_step(self, fold: int, progress: Progress) -> None:
        epoch_task = progress.add_task('Epoch:', total=self.config.epoch)

        def advance() -> None:
            progress.update(epoch_task, advance=1)

        self.train_config.ith_kfold = fold
        train_model(self.config, advance=advance)
        test_model(self.config)
        os.remove(self.config.model)
        progress.remove_task(epoch_task)
