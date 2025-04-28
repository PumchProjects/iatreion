from abc import ABC, abstractmethod

from rich.progress import track


class Trainer(ABC):
    @abstractmethod
    def train_step(self, fold: int) -> None: ...

    def train(self) -> None:
        for fold in track(range(5), description='Fold:'):
            self.train_step(fold)
