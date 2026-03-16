from typing import override

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from iatreion.configs import XgboostConfig
from iatreion.train_utils import TrainStepContext
from iatreion.utils import decode_string, encode_string, logger

from .base import Model
from .importance import ImportanceScore, calc_shap_importance


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
        self.param = config._param
        self.update_param()
        self.feature_types: list[str] = []

    def update_param(self) -> None:
        if self.num_class <= 2:
            self.param.update(
                {
                    'objective': 'binary:logistic',
                    'eval_metric': ['auc'],
                }
            )
        else:
            self.param.update(
                {
                    'objective': 'multi:softprob',
                    'num_class': self.num_class,
                }
            )

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
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
