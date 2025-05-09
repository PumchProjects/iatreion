from typing import override

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from iatreion.configs import CartConfig

from .base import ModelReturn, RawModel


class CartModel(RawModel):
    def __init__(self, config: CartConfig) -> None:
        super().__init__()
        self.config = config
        self.clf = DecisionTreeClassifier()

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(X, y)

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        y_pred = self.clf.predict(X)
        complexity = self.clf.get_depth()
        return y_pred, complexity
