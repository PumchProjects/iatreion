import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import shap
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from iatreion.configs import ImportanceMethod, ModelConfig, TrainConfig
from iatreion.train_utils import TrainStepContext
from iatreion.utils import decode_string, task

type ImportanceScore = dict[str, float]
type PredictProba = Callable[[NDArray], NDArray]


@dataclass(frozen=True)
class ShapBundle:
    values: NDArray[np.floating]
    base_values: NDArray[np.floating]
    data: NDArray[np.floating]
    y_true: NDArray[np.integer]
    sample_indices: NDArray[np.integer]
    feature_names: list[str]
    output_names: list[str]


def save_importance_score(
    train: TrainConfig,
    ctx: TrainStepContext,
    score: ImportanceScore,
    *,
    method: ImportanceMethod,
) -> None:
    if train._encode:
        score = {decode_string(name): value for name, value in score.items()}
    score_file = train._log_dir / ctx.get_importance_file(method)
    with score_file.open('w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=4)


def save_shap_bundle(
    train: TrainConfig,
    ctx: TrainStepContext,
    bundle: ShapBundle,
) -> None:
    np.savez_compressed(
        train._log_dir / ctx.shap_file,
        values=bundle.values,
        base_values=bundle.base_values,
        data=bundle.data,
        y_true=bundle.y_true,
        sample_indices=bundle.sample_indices,
        feature_names=np.asarray(bundle.feature_names, dtype=str),
        output_names=np.asarray(bundle.output_names, dtype=str),
    )


def _sample_importance_indices(
    n_samples: int,
    *,
    max_samples: int | None,
    seed: int,
) -> NDArray[np.integer]:
    if max_samples is None or n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def sample_importance_data(
    X: NDArray,
    y: NDArray,
    *,
    max_samples: int | None,
    seed: int,
) -> tuple[NDArray, NDArray]:
    index = _sample_importance_indices(
        X.shape[0],
        max_samples=max_samples,
        seed=seed,
    )
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
        config.train.num_class, y_sample, predict_proba(X_sample)
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
                    config.train.num_class, y_sample, predict_proba(permuted)
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


def _get_feature_names(train: TrainConfig, feature_names: list[str]) -> list[str]:
    if not train._encode:
        return feature_names
    return [decode_string(name) for name in feature_names]


def _get_output_names(train: TrainConfig, values: NDArray) -> list[str]:
    n_outputs = 1 if values.ndim == 2 else values.shape[-1]
    group_names = train._sorted_group_names
    if n_outputs == len(group_names):
        return group_names
    if n_outputs == 1 and len(group_names) == 2:
        return [group_names[-1]]
    if n_outputs == 1:
        return ['output_0']
    return [f'output_{index}' for index in range(n_outputs)]


def _build_shap_bundle(
    config: ModelConfig,
    ctx: TrainStepContext,
    explanation: shap.Explanation,
    sample_indices: NDArray[np.integer],
    y_true: NDArray[np.integer],
) -> ShapBundle:
    feature_names = _get_feature_names(config.train, list(ctx.db_enc.X_fname))
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    data = np.asarray(explanation.data, dtype=float)
    return ShapBundle(
        values=values,
        base_values=base_values,
        data=data,
        y_true=np.asarray(y_true, dtype=int).reshape(-1),
        sample_indices=np.asarray(sample_indices, dtype=int).reshape(-1),
        feature_names=feature_names,
        output_names=_get_output_names(config.train, values),
    )


def calc_shap_importance(
    config: ModelConfig,
    ctx: TrainStepContext,
    predict_proba: PredictProba | None = None,
    model: Any | None = None,
) -> ImportanceScore:
    X_test, y_test = ctx.test_data
    sample_indices = _sample_importance_indices(
        X_test.shape[0],
        max_samples=config.importance_max_samples,
        seed=config.train.seed,
    )
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    feature_names = _get_feature_names(config.train, list(ctx.db_enc.X_fname))
    if predict_proba is not None:
        explainer = shap.Explainer(
            predict_proba,
            X_sample,
            algorithm='permutation',
            feature_names=feature_names,
            output_names=config.train._sorted_group_names,
            seed=config.train.seed,
        )
    elif model is not None:
        explainer = shap.TreeExplainer(
            model,
            data=X_sample,
            model_output='probability',
            feature_names=feature_names,
        )
    else:
        raise ValueError('Either predict_proba or model must be provided.')

    explanation = explainer(X_sample)
    bundle = _build_shap_bundle(
        config,
        ctx,
        explanation,
        sample_indices,
        y_sample,
    )
    save_shap_bundle(config.train, ctx, bundle)

    importances = _reduce_shap_values(bundle.values, bundle.data.shape[1])
    return {
        name: float(importances[index])
        for index, name in enumerate(bundle.feature_names)
    }
