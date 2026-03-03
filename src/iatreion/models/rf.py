import json
from typing import override

from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from iatreion.configs import RandomForestConfig
from iatreion.rrl import TrainStepContext

from .base import Model, ModelReturn


class RandomForestModel(Model):
    def __init__(self, config: RandomForestConfig) -> None:
        super().__init__()
        self.config = config
        self.num_class = config.train.num_class
        self.forest = RandomForestClassifier(
            config.n_estimators,
            n_jobs=config.n_jobs,
            random_state=0,
        )

    @override
    def fit(self, X: NDArray, y: NDArray) -> None:
        self.forest.fit(X, y)

    def calc_importance(self, ctx: TrainStepContext) -> None:
        importances = self.forest.feature_importances_
        score = {
            name: importances[i].item() for i, name in enumerate(ctx.db_enc.X_fname)
        }
        score_file = (
            self.config.train._log_dir
            / f'score_{ctx.name}_{ctx.outer_fold}_{ctx.inner_fold}.json'
        )
        with score_file.open('w', encoding='utf-8') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)

    @override
    def predict(self, ctx: TrainStepContext, X: NDArray, y: NDArray) -> ModelReturn:
        y_score = self.forest.predict_proba(X)
        self.calc_importance(ctx)
        return y_score, {}
