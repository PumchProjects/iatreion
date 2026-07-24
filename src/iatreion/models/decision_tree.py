from typing import override

from sklearn.tree import DecisionTreeClassifier

from iatreion.configs import C45Config, CartConfig

from .rf import parse_max_features
from .sklearn_baseline import SklearnBaselineModel, parse_class_weight


class C45Model(SklearnBaselineModel):
    def __init__(self, config: C45Config) -> None:
        super().__init__(config)
        self.config: C45Config = config

    @override
    def _make_estimator(self) -> DecisionTreeClassifier:
        config = self.config
        return DecisionTreeClassifier(
            criterion=config.criterion,
            splitter=config.splitter,
            max_depth=None if config.max_depth == 0 else config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=parse_max_features(config.max_features),
            class_weight=parse_class_weight(config.class_weight),
            random_state=config.train.seed,
            ccp_alpha=config.ccp_alpha,
        )


class CartModel(SklearnBaselineModel):
    def __init__(self, config: CartConfig) -> None:
        super().__init__(config)
        self.config: CartConfig = config

    @override
    def _make_estimator(self) -> DecisionTreeClassifier:
        config = self.config
        return DecisionTreeClassifier(
            criterion='gini',
            splitter=config.splitter,
            max_depth=None if config.max_depth == 0 else config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=parse_max_features(config.max_features),
            class_weight=parse_class_weight(config.class_weight),
            random_state=config.train.seed,
            ccp_alpha=config.ccp_alpha,
        )
