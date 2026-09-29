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
from iatreion.utils import logger

from .base import Model
from .importance import (
    ImportanceScore,
    get_importance_sample,
    make_shap_bundle,
    save_shap_importance,
)

XGBOOST_MODEL_FILE = 'model.json'
XGBOOST_SHAP_RTOL = 1e-5
XGBOOST_SHAP_ATOL = 1e-5


def _validate_shap_additivity(
    contributions: NDArray,
    margins: NDArray,
) -> None:
    expected_margin_shape = contributions.shape[:2]
    if margins.shape != expected_margin_shape:
        raise ValueError(
            'Unexpected XGBoost raw-margin shape '
            f'{margins.shape}; expected {expected_margin_shape}. '
            f'Contribution dtype={contributions.dtype}, margin dtype={margins.dtype}.'
        )

    if not np.isfinite(contributions).all() or not np.isfinite(margins).all():
        raise ValueError(
            'XGBoost SHAP additivity inputs contain non-finite values. '
            f'Contribution shape={contributions.shape}, dtype={contributions.dtype}; '
            f'margin shape={margins.shape}, dtype={margins.dtype}.'
        )

    contribution_sums = contributions.sum(axis=2, dtype=contributions.dtype)
    if np.allclose(
        contribution_sums,
        margins,
        rtol=XGBOOST_SHAP_RTOL,
        atol=XGBOOST_SHAP_ATOL,
    ):
        return

    errors = np.abs(
        contribution_sums.astype(np.float64) - margins.astype(np.float64)
    )
    raise ValueError(
        'XGBoost SHAP contributions do not add up to raw margins. '
        f'Contribution shape={contributions.shape}, dtype={contributions.dtype}; '
        f'margin shape={margins.shape}, dtype={margins.dtype}; '
        f'max_abs_error={float(errors.max()):.9g}; '
        f'rtol={XGBOOST_SHAP_RTOL:g}, atol={XGBOOST_SHAP_ATOL:g}.'
    )


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
        score = self.bst.get_score(importance_type='gain')
        return {
            name: float(score.get(f'f{index}', 0.0))
            for index, name in enumerate(ctx.db_enc.X_fname)
        }

    @override
    def _calc_shap_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        sample = get_importance_sample(self.config, ctx)
        dtest = xgb.DMatrix(
            sample.X,
            feature_types=self.feature_types,
            enable_categorical=True,
        )
        contributions = np.asarray(
            self.bst.predict(
                dtest,
                pred_contribs=True,
                strict_shape=True,
            )
        )
        if contributions.ndim != 3 or contributions.shape[2] != sample.X.shape[1] + 1:
            raise ValueError(
                'Unexpected XGBoost SHAP contribution shape '
                f'{contributions.shape}; expected (samples, outputs, features + 1).'
            )

        margins = np.asarray(
            self.bst.predict(
                dtest,
                output_margin=True,
                strict_shape=True,
            )
        )
        _validate_shap_additivity(contributions, margins)
        contributions = contributions.astype(float, copy=False)

        values = np.transpose(contributions[:, :, :-1], (0, 2, 1))
        base_values = contributions[:, :, -1]
        if values.shape[2] == 1:
            values = values[:, :, 0]
            base_values = base_values[:, 0]
        bundle = make_shap_bundle(
            self.config,
            ctx,
            values=values,
            base_values=base_values,
            data=sample.X,
            y_true=sample.y,
            sample_indices=sample.indices,
            explainer='xgboost-pred-contribs',
            output_space='raw-margin',
            data_scope=sample.data_scope,
        )
        return save_shap_importance(self.config, ctx, bundle)
