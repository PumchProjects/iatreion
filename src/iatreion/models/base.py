from abc import ABC, abstractmethod

import pandas as pd
from numpy.typing import NDArray

from iatreion.rrl import TrainStepContext

type ModelReturn = tuple[NDArray, dict[str, float | tuple[float, str]]]


class Model(ABC):
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> None: ...

    @abstractmethod
    def predict(self, ctx: TrainStepContext, X: NDArray, y: NDArray) -> ModelReturn: ...


class RawModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn: ...
