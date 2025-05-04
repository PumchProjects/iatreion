import os
from typing import override

from rich.progress import Progress

from iatreion.configs import RrlConfig
from iatreion.models import ModelReturn
from iatreion.rrl import Samples
from iatreion.rrl.experiment import test_model, train_model

from .base import Trainer


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__(config.dataset, config.train)
        self.config = config

    @override
    def train_step(self, samples: Samples, progress: Progress) -> ModelReturn:
        epoch_task = progress.add_task('Epoch:', total=self.config.epoch)

        def advance() -> None:
            progress.update(epoch_task, advance=1)

        train_model(self.config, samples, advance=advance)
        y_score, complexity = test_model(self.config, samples)
        os.remove(self.config.model)
        progress.remove_task(epoch_task)
        return y_score, complexity
