from pathlib import Path
from typing import Any, override

import joblib
import numpy as np
from numpy.typing import NDArray

from iatreion.configs import ModelConfig
from iatreion.train_utils import TrainStepContext
from iatreion.train_utils.artifacts import (
    get_artifact_dir,
    get_transform_artifact_path,
)
from iatreion.train_utils.preprocessing import DBEncoderArtifact

from .base import Model
from .importance import ImportanceScore, calc_shap_importance

SKLEARN_MODEL_FILE = 'model.joblib'


def parse_class_weight(value: str) -> str | None:
    return None if value == 'none' else value


class SklearnBaselineModel(Model):
    estimator: Any

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_class = config.train.num_class
        self.estimator = self._make_estimator()

    def _make_estimator(self) -> Any:
        raise NotImplementedError

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self.estimator = self._make_estimator()
        self.estimator.fit(X, y)

    @override
    def save_final(self, ctx: TrainStepContext) -> None:
        artifact_dir = get_artifact_dir(self.config.train._log_dir, ctx.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ctx.db_enc.save_transform_artifact(
            get_transform_artifact_path(self.config.train._log_dir, ctx.name)
        )
        joblib.dump(self.estimator, artifact_dir / SKLEARN_MODEL_FILE)

    @override
    def load_final(self, artifact_dir: Path, transform: DBEncoderArtifact) -> None:
        self.estimator = joblib.load(artifact_dir / SKLEARN_MODEL_FILE)

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        return self.estimator.predict_proba(X)

    @override
    def _calc_native_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        estimator = self.estimator
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = np.asarray(estimator.coef_)
            importances = np.abs(coef).mean(axis=0)
        else:
            raise NotImplementedError
        return {
            name: float(importances[i]) for i, name in enumerate(ctx.db_enc.X_fname)
        }

    @override
    def _calc_shap_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        return calc_shap_importance(self.config, ctx, model=self.estimator)
