from typing import override

from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from iatreion.configs import RandomForestConfig
from iatreion.train_utils import TrainStepContext

from .base import Model
from .importance import ImportanceScore, calc_shap_importance


class RandomForestModel(Model):
    def __init__(self, config: RandomForestConfig) -> None:
        super().__init__()
        self.config: RandomForestConfig = config
        self.num_class = config.train.num_class
        self.forest = RandomForestClassifier(
            config.n_estimators,
            n_jobs=config.n_jobs,
            random_state=0,
        )

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self.forest.fit(X, y)

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        return self.forest.predict_proba(X)

    @override
    def _calc_native_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        importances = self.forest.feature_importances_
        return {
            name: float(importances[i]) for i, name in enumerate(ctx.db_enc.X_fname)
        }

    @override
    def _calc_shap_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        return calc_shap_importance(self.config, ctx, model=self.forest)
