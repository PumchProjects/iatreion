from pathlib import Path
from typing import override

import joblib
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from iatreion.configs import RandomForestConfig
from iatreion.train_utils import TrainStepContext
from iatreion.train_utils.artifacts import (
    get_artifact_dir,
    get_transform_artifact_path,
)
from iatreion.train_utils.preprocessing import DBEncoderArtifact

from .base import Model
from .importance import ImportanceScore, calc_shap_importance

RANDOM_FOREST_MODEL_FILE = 'model.joblib'


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
    def save_final(self, ctx: TrainStepContext) -> None:
        artifact_dir = get_artifact_dir(self.config.train._log_dir, ctx.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ctx.db_enc.save_transform_artifact(
            get_transform_artifact_path(self.config.train._log_dir, ctx.name)
        )
        joblib.dump(self.forest, artifact_dir / RANDOM_FOREST_MODEL_FILE)

    @override
    def load_final(self, artifact_dir: Path, transform: DBEncoderArtifact) -> None:
        self.forest = joblib.load(artifact_dir / RANDOM_FOREST_MODEL_FILE)

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
