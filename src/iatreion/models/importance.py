import json
from collections.abc import Callable
from typing import Any

import numpy as np
import shap
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from iatreion.configs import ImportanceMethod, ModelConfig, TrainConfig
from iatreion.rrl import TrainStepContext
from iatreion.utils import decode_string, task

type ImportanceScore = dict[str, float]
type PredictProba = Callable[[NDArray, NDArray], NDArray]


def save_importance_score(
    train: TrainConfig,
    ctx: TrainStepContext,
    score: ImportanceScore,
    *,
    method: ImportanceMethod,
) -> None:
    if train._encode:
        score = {decode_string(name): value for name, value in score.items()}
    score_file = train.get_importance_file(
        ctx.name,
        ctx.outer_fold,
        ctx.inner_fold,
        method=method,
    )
    with score_file.open('w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=4)


def sample_importance_data(
    X: NDArray,
    y: NDArray,
    *,
    max_samples: int | None,
    seed: int,
) -> tuple[NDArray, NDArray]:
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    index = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[index], y[index]


def _calc_auc_score(num_class: int, y_true: NDArray, y_score: NDArray) -> float:
    try:
        if num_class <= 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                y_score,
                average='macro',
                multi_class='ovr',
                labels=list(range(num_class)),
            )
        )
    except ValueError:
        return np.nan


def calc_permutation_importance(
    config: ModelConfig,
    ctx: TrainStepContext,
    predict_proba: PredictProba,
) -> ImportanceScore:
    X_sample, y_sample = sample_importance_data(
        *ctx.test_data,
        max_samples=config.importance_max_samples,
        seed=config.train.seed,
    )
    baseline = _calc_auc_score(
        config.train.num_class, y_sample, predict_proba(X_sample, y_sample)
    )
    feature_names = ctx.db_enc.X_fname
    if np.isnan(baseline):
        return {name: np.nan for name in feature_names}

    rng = np.random.default_rng(config.train.seed)
    repeats = max(1, config.importance_repeats)
    score: ImportanceScore = {}
    with task('Permutation:', len(feature_names) * repeats) as permutation_task:
        for idx, name in enumerate(feature_names):
            deltas: list[float] = []
            for _ in range(repeats):
                permuted = X_sample.copy()
                permuted[:, idx] = rng.permutation(permuted[:, idx])
                auc = _calc_auc_score(
                    config.train.num_class, y_sample, predict_proba(permuted, y_sample)
                )
                if np.isnan(auc):
                    continue
                deltas.append(baseline - auc)
                permutation_task()
            score[name] = float(np.mean(deltas)) if deltas else np.nan
    return score


def _reduce_shap_values(values: object, n_features: int) -> NDArray:
    if isinstance(values, list):
        arr = np.stack([np.asarray(value) for value in values], axis=0)
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return np.abs(arr).mean(axis=(0, 2))
        if arr.shape[2] == n_features:
            return np.abs(arr).mean(axis=(0, 1))
    raise ValueError(
        f'Unsupported SHAP shape {arr.shape}; '
        f'expected (*, {n_features})-compatible array.'
    )


def calc_tree_shap_importance(
    config: ModelConfig,
    ctx: TrainStepContext,
    model: Any,
) -> ImportanceScore:
    X_sample, _ = sample_importance_data(
        *ctx.test_data,
        max_samples=config.importance_max_samples,
        seed=config.train.seed,
    )
    feature_names = ctx.db_enc.X_fname
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    importances = _reduce_shap_values(shap_values, X_sample.shape[1])
    return {name: float(importances[i]) for i, name in enumerate(feature_names)}
