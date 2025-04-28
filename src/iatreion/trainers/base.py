from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train_step(self, fold: int) -> None: ...

    def train(self) -> None:
        for fold in range(5):
            self.train_step(fold)
