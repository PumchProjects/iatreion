from typing import override

import pandas as pd

from iatreion.configs import GatreeConfig
from iatreion.gatree import GATreeClassifier
from iatreion.utils import logger

from .base import ModelReturn, RawModel


class GatreeModel(RawModel):
    def __init__(self, config: GatreeConfig) -> None:
        super().__init__()
        self.config = config
        self.gatree = GATreeClassifier(max_depth=config.max_depth, n_jobs=config.n_jobs)

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.gatree.fit(
            X,
            y,
            population_size=self.config.population_size,
            max_iter=self.config.max_iter,
        )
        if self.config.plot:
            logger.info('[bold green]Tree', extra={'markup': True})
            self.gatree.plot()

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        y_pred = self.gatree.predict(X)
        depth = self.gatree._tree.max_depth() - 1
        n_leaves = len(self.gatree._tree.get_leaves())
        return y_pred, {'Depth': depth, '#Leaf': n_leaves}
