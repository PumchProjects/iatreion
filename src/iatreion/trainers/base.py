from abc import ABC, abstractmethod

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn


class Trainer(ABC):
    @abstractmethod
    def train_step(self, fold: int, progress: Progress) -> None: ...

    def train(self) -> None:
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
        ) as progress:
            fold_task = progress.add_task('Fold:', total=5)
            for fold in range(5):
                self.train_step(fold, progress)
                progress.update(fold_task, advance=1)
