from pathlib import Path
from typing import override

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from iatreion.configs import XgboostConfig
from iatreion.train_utils import TrainStepContext
from iatreion.train_utils.artifacts import (
    get_artifact_dir,
    get_transform_artifact_path,
)
from iatreion.train_utils.preprocessing import DBEncoderArtifact
from iatreion.utils import decode_string, encode_string, logger

from .base import Model
from .importance import ImportanceScore, calc_shap_importance

XGBOOST_MODEL_FILE = 'model.json'


class XgbLogging(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        log_list = [f'[{epoch}]']
        for data, metric in evals_log.items():
            for m_key, m_value in metric.items():
                log_list.append(f'{data}-{m_key}:{m_value[-1]:.5f}')
        logger.info('\t'.join(log_list))
        return False


class XgboostModel(Model):
    def __init__(self, config: XgboostConfig) -> None:
        super().__init__()
        self.config: XgboostConfig = config
        self.num_class = config.train.num_class
        self.param: dict[str, object] = {}
        self.feature_types: list[str] = []

    def _params(self) -> dict[str, object]:
        config = self.config
        params: dict[str, object] = {
            'device': config.device,
            'tree_method': config.tree_method,
            'learning_rate': config.learning_rate,
            'max_depth': config.max_depth,
            'min_child_weight': config.min_child_weight,
            'subsample': config.subsample,
            'colsample_bytree': config.colsample_bytree,
            'gamma': config.gamma,
            'reg_lambda': config.reg_lambda,
            'reg_alpha': config.reg_alpha,
            'seed': config.train.seed,
        }
        if self.num_class <= 2:
            params |= {
                'objective': 'binary:logistic',
                'eval_metric': ['auc'],
                'scale_pos_weight': config.scale_pos_weight,
            }
        else:
            params |= {
                'objective': 'multi:softprob',
                'num_class': self.num_class,
            }
        return params

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self.param = self._params()
        dtrain = xgb.DMatrix(
            X,
            y,
            feature_types=self.feature_types,
            enable_categorical=True,
        )
        self.bst = xgb.train(
            self.param,
            dtrain,
            self.config.num_round,
            evals=[(dtrain, 'train')],
            verbose_eval=False,
            callbacks=[XgbLogging()],
        )

    @override
    def fit(self, ctx: TrainStepContext) -> None:
        self.feature_types = [
            *('i' for _ in range(ctx.db_enc.binary_flen)),
            *('c' for _ in range(ctx.db_enc.categorical_flen)),
            *('q' for _ in range(ctx.db_enc.numeric_flen)),
        ]
        super().fit(ctx)

    @override
    def save_final(self, ctx: TrainStepContext) -> None:
        artifact_dir = get_artifact_dir(self.config.train._log_dir, ctx.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ctx.db_enc.save_transform_artifact(
            get_transform_artifact_path(self.config.train._log_dir, ctx.name)
        )
        self.bst.save_model(artifact_dir / XGBOOST_MODEL_FILE)

    @override
    def load_final(self, artifact_dir: Path, transform: DBEncoderArtifact) -> None:
        self.feature_types = transform.feature_types
        self.bst = xgb.Booster()
        self.bst.load_model(artifact_dir / XGBOOST_MODEL_FILE)

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        dtest = xgb.DMatrix(X, feature_types=self.feature_types)
        y_score = self.bst.predict(dtest)
        if self.num_class <= 2:
            return np.stack([1 - y_score, y_score], axis=-1)
        return y_score.reshape(X.shape[0], -1)

    @override
    def _calc_native_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        fmap_file = (
            self.config.train._log_dir
            / f'fmap_{ctx.name}_{ctx.outer_fold}_{ctx.inner_fold}.tsv'
        )
        with fmap_file.open('w', encoding='utf-8') as f:
            for i, (name, ftype) in enumerate(
                zip(ctx.db_enc.X_fname, self.feature_types, strict=True)
            ):
                f.write(f'{i}\t{encode_string(name, " ")}\t{ftype}\n')
        raw_score = self.bst.get_score(str(fmap_file), importance_type='gain')
        score = {decode_string(name): float(value) for name, value in raw_score.items()}
        return {name: score.get(name, 0.0) for name in ctx.db_enc.X_fname}

    @override
    def _calc_shap_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        return calc_shap_importance(self.config, ctx, model=self.bst)
