import json
from typing import override

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from iatreion.configs import XgboostConfig
from iatreion.rrl import TrainStepContext
from iatreion.utils import decode_string, logger

from .base import Model, ModelReturn


class XgbLogging(xgb.callback.TrainingCallback):
    def __init__(self):
        pass

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
        self.config = config
        self.num_class = config.train.num_class
        self.param = config.param
        self.update_param()

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
    def fit(self, X: NDArray, y: NDArray) -> None:
        dtrain = xgb.DMatrix(X, y)
        evals = [(dtrain, 'train')]
        callbacks = [XgbLogging()]
        self.bst = xgb.train(
            self.param,
            dtrain,
            self.config.num_round,
            evals=evals,
            verbose_eval=False,
            callbacks=callbacks,
        )

    def calc_importance(self, ctx: TrainStepContext) -> None:
        fmap_file = (
            self.config.train.log_dir
            / f'fmap_{ctx.name}_{ctx.outer_fold}_{ctx.inner_fold}.tsv'
        )
        with fmap_file.open('w', encoding='utf-8') as f:
            for i, name in enumerate(ctx.db_enc.X_fname):
                type = 'i' if i < ctx.db_enc.discrete_flen else 'q'
                f.write(f'{i}\t{name}\t{type}\n')
        score = {decode_string(f): v for f, v in self.bst.get_fscore(fmap_file).items()}
        score_file = (
            self.config.train.log_dir
            / f'{ctx.name}_{ctx.outer_fold}_{ctx.inner_fold}.json'
        )
        with score_file.open('w', encoding='utf-8') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)

    @override
    def predict(self, ctx: TrainStepContext, X: NDArray, y: NDArray) -> ModelReturn:
        dtest = xgb.DMatrix(X, y)
        y_score = self.bst.predict(dtest)
        if self.num_class <= 2:
            y_score = np.stack([1 - y_score, y_score], axis=-1)
        else:
            y_score = y_score.reshape(X.shape[0], -1)
        self.calc_importance(ctx)
        return y_score, {}
