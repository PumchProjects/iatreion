from abc import ABC, abstractmethod

from numpy.typing import NDArray

type ModelReturn = tuple[NDArray, float]


class Model(ABC):
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> None: ...

    @abstractmethod
    def predict(self, X: NDArray, y: NDArray) -> ModelReturn: ...
