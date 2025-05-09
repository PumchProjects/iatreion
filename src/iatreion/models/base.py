from abc import ABC, abstractmethod

import pandas as pd
from numpy.typing import ArrayLike, NDArray

type ModelReturn = tuple[ArrayLike, dict[str, float]]


class Model(ABC):
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> None: ...

    @abstractmethod
    def predict(self, X: NDArray, y: NDArray) -> ModelReturn: ...


class RawModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn: ...
