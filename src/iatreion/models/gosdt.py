from typing import Protocol, override

import pandas as pd
from gosdt import GOSDTClassifier, NumericBinarizer, ThresholdGuessBinarizer
from gosdt._tree import Leaf, Node
from sklearn.ensemble import GradientBoostingClassifier

from iatreion.configs import GosdtConfig

from .base import ModelReturn, RawModel


class Collector[T](Protocol):
    def __call__(self, left: T = ..., right: T = ..., /) -> T: ...


def collect_depth(left: int = -1, right: int = -1) -> int:
    return max(left, right) + 1


def traverse[T](node: Node | Leaf, collect: Collector[T]) -> T:
    if isinstance(node, Node):
        left = traverse(node.left_child, collect)
        right = traverse(node.right_child, collect)
        return collect(left, right)
    elif isinstance(node, Leaf):
        return collect()


class GosdtModel(RawModel):
    def __init__(self, config: GosdtConfig) -> None:
        super().__init__()
        self.config = config
        self.tgb = ThresholdGuessBinarizer(
            n_estimators=config.gbdt_n_est,
            max_depth=config.gbdt_max_depth,
            random_state=2021,
        ).set_output(transform='pandas')
        self.nb = NumericBinarizer().set_output(transform='pandas')
        self.clf = GOSDTClassifier(
            regularization=config.regularization,
            similar_support=config.similar_support,
            time_limit=config.time_limit,
            depth_budget=config.depth_budget,
            verbose=config.verbose,
        )

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.config.guess_th:
            X_bin = self.tgb.fit_transform(X, y)
        else:
            X_bin = self.nb.fit_transform(X, y)

        if self.config.guess_lb:
            enc = GradientBoostingClassifier(
                n_estimators=self.config.gbdt_n_est,
                max_depth=self.config.gbdt_max_depth,
                random_state=42,
            )
            enc.fit(X_bin, y)
            warm_labels = enc.predict(X_bin)
            self.clf.fit(X_bin, y, y_ref=warm_labels)
        else:
            self.clf.fit(X_bin, y)

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        X_bin = self.tgb.transform(X) if self.config.guess_th else self.nb.transform(X)

        y_pred = self.clf.predict(X_bin)
        complexity = traverse(self.clf.trees_[0].tree, collect_depth)
        return y_pred, complexity
