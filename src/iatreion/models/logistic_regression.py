from typing import override

from sklearn.linear_model import LogisticRegression

from iatreion.configs import LogisticRegressionConfig

from .sklearn_baseline import SklearnBaselineModel, parse_class_weight


class LogisticRegressionModel(SklearnBaselineModel):
    def __init__(self, config: LogisticRegressionConfig) -> None:
        super().__init__(config)
        self.config: LogisticRegressionConfig = config

    @override
    def _make_estimator(self) -> LogisticRegression:
        config = self.config
        return LogisticRegression(
            penalty=None if config.penalty == 'none' else config.penalty,
            C=config.C,
            solver=config.solver,
            max_iter=config.max_iter,
            tol=config.tol,
            class_weight=parse_class_weight(config.class_weight),
            random_state=config.train.seed,
        )
