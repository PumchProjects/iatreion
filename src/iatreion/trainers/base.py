from abc import ABC, abstractmethod

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

from iatreion.configs import TrainConfig


class Trainer(ABC):
    def __init__(self, config: TrainConfig) -> None:
        self.train_config = config

    @abstractmethod
    def train_step(self, fold: int, progress: Progress) -> None: ...

    def train(self) -> None:
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
        ) as progress:
            fold_task = progress.add_task('Fold:', total=self.train_config.n_splits)
            for fold in range(self.train_config.n_splits):
                self.train_step(fold, progress)
                progress.update(fold_task, advance=1)
