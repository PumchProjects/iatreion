import json
import multiprocessing as mp
import signal
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

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
    explainer: str
    output_space: str
    data_scope: str


@dataclass(frozen=True)
class ImportanceSample:
    X: NDArray
    y: NDArray
    indices: NDArray[np.integer]
    data_scope: str


class TreeShapWorkerError(RuntimeError):
    pass


def save_importance_score(
    config: ModelConfig,
    ctx: TrainStepContext,
    score: ImportanceScore,
    *,
    method: ImportanceMethod,
) -> None:
    if config.dataset._encode:
        score = {decode_string(name): value for name, value in score.items()}
    train = config.train
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
        explainer=np.asarray(bundle.explainer, dtype=str),
        output_space=np.asarray(bundle.output_space, dtype=str),
        data_scope=np.asarray(bundle.data_scope, dtype=str),
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


def get_importance_sample(
    config: ModelConfig,
    ctx: TrainStepContext,
) -> ImportanceSample:
    data_scope = 'training' if config.train.final else 'held-out'
    X, y = ctx.train_data if config.train.final else ctx.test_data
    if X.shape[0] == 0:
        raise ValueError(f'No {data_scope} samples available for importance.')
    indices = _sample_importance_indices(
        X.shape[0],
        max_samples=config.importance_max_samples,
        seed=config.train.seed,
    )
    return ImportanceSample(
        X=X[indices],
        y=y[indices],
        indices=indices,
        data_scope=data_scope,
    )


def _calc_auroc_score(num_class: int, y_true: NDArray, y_score: NDArray) -> float:
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
    sample = get_importance_sample(config, ctx)
    X_sample, y_sample = sample.X, sample.y
    baseline = _calc_auroc_score(
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
                auroc = _calc_auroc_score(
                    config.train.num_class, y_sample, predict_proba(permuted)
                )
                if np.isnan(auroc):
                    continue
                deltas.append(baseline - auroc)
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


def _get_feature_names(config: ModelConfig, feature_names: list[str]) -> list[str]:
    if not config.dataset._encode:
        return feature_names
    return [decode_string(name) for name in feature_names]


def _get_output_names(train: TrainConfig, values: NDArray) -> list[str]:
    n_outputs = 1 if values.ndim == 2 else values.shape[-1]
    group_names = train.group_labels
    if n_outputs == len(group_names):
        return group_names
    if n_outputs == 1 and len(group_names) == 2:
        return [group_names[-1]]
    if n_outputs == 1:
        return ['output_0']
    return [f'output_{index}' for index in range(n_outputs)]


def make_shap_bundle(
    config: ModelConfig,
    ctx: TrainStepContext,
    *,
    values: NDArray,
    base_values: NDArray,
    data: NDArray,
    y_true: NDArray,
    sample_indices: NDArray[np.integer],
    explainer: str,
    output_space: str,
    data_scope: str,
) -> ShapBundle:
    feature_names = _get_feature_names(config, list(ctx.db_enc.X_fname))
    values = np.asarray(values, dtype=float)
    return ShapBundle(
        values=values,
        base_values=np.asarray(base_values, dtype=float),
        data=np.asarray(data, dtype=float),
        y_true=np.asarray(y_true, dtype=int).reshape(-1),
        sample_indices=np.asarray(sample_indices, dtype=int).reshape(-1),
        feature_names=feature_names,
        output_names=_get_output_names(config.train, values),
        explainer=explainer,
        output_space=output_space,
        data_scope=data_scope,
    )


def save_shap_importance(
    config: ModelConfig,
    ctx: TrainStepContext,
    bundle: ShapBundle,
) -> ImportanceScore:
    save_shap_bundle(config.train, ctx, bundle)
    importances = _reduce_shap_values(bundle.values, bundle.data.shape[1])
    return {
        name: float(importances[index])
        for index, name in enumerate(bundle.feature_names)
    }


def calc_shap_importance(
    config: ModelConfig,
    ctx: TrainStepContext,
    predict_proba: PredictProba,
    *,
    explainer_name: str = 'permutation',
) -> ImportanceScore:
    sample = get_importance_sample(config, ctx)
    feature_names = _get_feature_names(config, list(ctx.db_enc.X_fname))
    explainer = shap.Explainer(
        predict_proba,
        sample.X,
        algorithm='permutation',
        feature_names=feature_names,
        output_names=config.train.group_labels,
        seed=config.train.seed,
    )
    explanation = explainer(sample.X)
    bundle = make_shap_bundle(
        config,
        ctx,
        values=explanation.values,
        base_values=explanation.base_values,
        data=sample.X,
        y_true=sample.y,
        sample_indices=sample.indices,
        explainer=explainer_name,
        output_space='probability',
        data_scope=sample.data_scope,
    )
    return save_shap_importance(config, ctx, bundle)


def _random_forest_tree_shap_worker(
    model: object,
    X: NDArray,
    feature_names: list[str],
    result_path: str,
    error_path: str,
) -> None:
    try:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation='tree_path_dependent',
            model_output='raw',
            feature_names=feature_names,
        )
        explanation = explainer(X)
        np.savez_compressed(
            result_path,
            values=np.asarray(explanation.values, dtype=float),
            base_values=np.asarray(explanation.base_values, dtype=float),
        )
    except BaseException:
        Path(error_path).write_text(traceback.format_exc(), encoding='utf-8')
        raise


def _run_random_forest_tree_shap(
    model: object,
    X: NDArray,
    feature_names: list[str],
) -> tuple[NDArray, NDArray]:
    with TemporaryDirectory(prefix='iatreion-rf-shap-') as tmp:
        root = Path(tmp)
        result_path = root / 'result.npz'
        error_path = root / 'error.txt'
        process = mp.get_context('spawn').Process(
            target=_random_forest_tree_shap_worker,
            args=(model, X, feature_names, str(result_path), str(error_path)),
        )
        process.start()
        process.join()
        exitcode = process.exitcode
        process.close()
        if exitcode != 0:
            if exitcode is not None and exitcode < 0:
                try:
                    reason = signal.Signals(-exitcode).name
                except ValueError:
                    reason = f'signal {-exitcode}'
            else:
                reason = f'exit code {exitcode}'
            detail = (
                error_path.read_text(encoding='utf-8').strip()
                if error_path.exists()
                else 'no Python traceback (the worker may have crashed in native code)'
            )
            raise TreeShapWorkerError(
                f'Random Forest TreeSHAP worker failed with {reason}: {detail}'
            )
        if not result_path.exists():
            raise TreeShapWorkerError(
                'Random Forest TreeSHAP worker exited without a result.'
            )
        with np.load(result_path) as arrays:
            values = np.asarray(arrays['values'], dtype=float)
            base_values = np.asarray(arrays['base_values'], dtype=float)
    return values, base_values


def calc_random_forest_shap_importance(
    config: ModelConfig,
    ctx: TrainStepContext,
    model: object,
) -> ImportanceScore:
    sample = get_importance_sample(config, ctx)
    feature_names = _get_feature_names(config, list(ctx.db_enc.X_fname))
    values, base_values = _run_random_forest_tree_shap(
        model,
        sample.X,
        feature_names,
    )
    bundle = make_shap_bundle(
        config,
        ctx,
        values=values,
        base_values=base_values,
        data=sample.X,
        y_true=sample.y,
        sample_indices=sample.indices,
        explainer='tree-path-dependent',
        output_space='probability',
        data_scope=sample.data_scope,
    )
    return save_shap_importance(config, ctx, bundle)
