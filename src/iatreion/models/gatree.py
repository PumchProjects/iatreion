from typing import override

import pandas as pd

from iatreion.configs import GatreeConfig
from iatreion.gatree import GATreeClassifier

from .base import ModelReturn, RawModel


class GatreeModel(RawModel):
    def __init__(self, config: GatreeConfig) -> None:
        super().__init__()
        self.config = config
        self.gatree = GATreeClassifier(n_jobs=config.n_jobs)

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.gatree.fit(
            X,
            y,
            population_size=self.config.population_size,
            max_iter=self.config.max_iter,
        )

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        y_pred = self.gatree.predict(X)
        return y_pred, self.gatree._tree.max_depth() - 1
