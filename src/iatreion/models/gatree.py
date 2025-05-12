from typing import override

import pandas as pd
from sklearn.metrics import accuracy_score

from iatreion.configs import GatreeConfig
from iatreion.gatree import GATreeClassifier
from iatreion.utils import logger

from .base import ModelReturn, RawModel


def fitness_function(root):
    return 1 - accuracy_score(root.y_true, root.y_pred) + 0.002 * root.n_leaves()


class GatreeModel(RawModel):
    def __init__(self, config: GatreeConfig) -> None:
        super().__init__()
        self.config = config
        self.gatree = GATreeClassifier(max_depth=config.max_depth, fitness_function=fitness_function, n_jobs=config.n_jobs)

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.gatree.fit(
            X,
            y,
            population_size=self.config.population_size,
            max_iter=self.config.max_iter,
            mutation_probability=self.config.mutation_probability,
            elite_size=self.config.elite_size,
            selection_tournament_size=self.config.selection_tournament_size,
        )
        if self.config.plot:
            logger.info('[bold green]Tree', extra={'markup': True})
            self.gatree.plot()

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        y_pred = self.gatree.predict(X)
        depth = self.gatree._tree.max_depth() - 1
        n_leaves = self.gatree._tree.n_leaves()
        return y_pred, {'Depth': depth, '#Leaf': n_leaves, 'Fit': 1 - self.gatree._tree.fitness}
