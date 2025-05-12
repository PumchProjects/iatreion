from typing import override

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

from iatreion.configs import CartConfig
from iatreion.utils import logger

from .base import ModelReturn, RawModel


class CartModel(RawModel):
    def __init__(self, config: CartConfig) -> None:
        super().__init__()
        self.config = config
        self.clf = DecisionTreeClassifier(max_depth=config.max_depth)
        self.objective: float | None = None

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(X, y)
        if self.config.plot:
            plot = export_text(self.clf, feature_names=X.columns)
            logger.info('[bold green]Tree', extra={'markup': True})
            logger.info(plot)
        y_pred, complexity = self._predict(X)
        self.objective = accuracy_score(y, y_pred) - 0.002 * complexity['#Leaf']

    def _predict(self, X: pd.DataFrame) -> ModelReturn:
        y_pred = self.clf.predict(X)
        depth = self.clf.get_depth()
        n_leaves = self.clf.get_n_leaves()
        return y_pred, {'Depth': depth, '#Leaf': n_leaves}
    
    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        y_pred, complexity = self._predict(X)
        assert self.objective is not None
        return y_pred, complexity | {'Fit': self.objective}
