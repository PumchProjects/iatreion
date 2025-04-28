from typing import override

from rich.progress import Progress

from iatreion.rrl import RrlConfig, test_model, train_model

from .base import Trainer


class RrlTrainer(Trainer):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__()
        self.config = config

    @override
    def train_step(self, fold: int, progress: Progress) -> None:
        epoch_task = progress.add_task('Epoch:', total=self.config.epoch)

        def advance() -> None:
            progress.update(epoch_task, advance=1)

        self.config.ith_kfold = fold
        train_model(self.config, advance=advance)
        test_model(self.config)
