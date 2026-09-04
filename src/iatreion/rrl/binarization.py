from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import shap
import torch
from numpy.typing import NDArray
from sklearn.utils import resample

MAX_SAMPLE_SIZE = 256
FEATURE_GROUP_SHIFTS = (1, 2, 4)

type JumpCandidates = tuple[NDArray[np.float64], NDArray[np.float64]]
type QuantileCandidates = tuple[NDArray[np.float64], NDArray[np.float64]]


def _make_classifier(model_path: Path, random_state: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(model_path),
        n_estimators=1,
        auto_scale_n_estimators=False,
        random_state=random_state,
        n_preprocessing_jobs=1,
    )


def _make_attention_classifier(model_path: Path, random_state: int):
    from tabpfn import TabPFNClassifier
    from tabpfn.preprocessing import PreprocessorConfig

    return TabPFNClassifier(
        model_path=str(model_path),
        n_estimators=1,
        auto_scale_n_estimators=False,
        random_state=random_state,
        n_preprocessing_jobs=1,
        inference_config={
            'PREPROCESS_TRANSFORMS': [
                PreprocessorConfig(
                    'none',
                    categorical_name='numeric',
                    max_features_per_estimator=2000,
                )
            ],
            'FEATURE_SHIFT_METHOD': None,
            'CLASS_SHIFT_METHOD': None,
            'FINGERPRINT_FEATURE': False,
            'POLYNOMIAL_FEATURES': 'no',
            'OUTLIER_REMOVAL_STD': None,
            'ENABLE_GPU_PREPROCESSING': False,
        },
    )


def _sample_indices(y: NDArray, random_state: int) -> NDArray[np.integer]:
    indices = np.arange(len(y))
    if len(indices) <= MAX_SAMPLE_SIZE:
        return indices
    return np.sort(
        resample(
            indices,
            replace=False,
            n_samples=MAX_SAMPLE_SIZE,
            stratify=y,
            random_state=random_state,
        )
    )


def _explain_sample(
    classifier,
    X: NDArray,
    y: NDArray,
    random_state: int,
) -> tuple[NDArray, NDArray]:
    X_sample = X[_sample_indices(y, random_state)]
    explainer = shap.Explainer(
        classifier.predict_proba,
        X_sample,
        algorithm='permutation',
        seed=random_state,
    )
    explanation = explainer(
        X_sample,
        max_evals=2 * X.shape[1] + 1,
        silent=True,
    )
    return X_sample, np.asarray(explanation.values)


def _shap_jump_candidates(
    X: NDArray,
    shap_values: NDArray,
) -> list[JumpCandidates]:
    X = np.asarray(X, dtype=np.float64)
    values = np.asarray(shap_values, dtype=np.float64)
    values = values.reshape(len(X), X.shape[1], -1)
    result: list[JumpCandidates] = []

    for column in range(X.shape[1]):
        x = X[:, column]
        feature_values = values[:, column]
        observed = np.isfinite(x) & np.all(np.isfinite(feature_values), axis=1)
        x = x[observed]
        feature_values = feature_values[observed]
        unique, inverse, counts = np.unique(
            x,
            return_inverse=True,
            return_counts=True,
        )
        if len(unique) < 2:
            empty = np.empty(0, dtype=np.float64)
            result.append((empty, empty))
            continue

        means = np.zeros((len(unique), feature_values.shape[1]), dtype=np.float64)
        np.add.at(means, inverse, feature_values)
        means /= counts[:, None]
        scores = np.linalg.norm(np.diff(means, axis=0), axis=1)
        candidates = unique[:-1] + (unique[1:] - unique[:-1]) / 2
        valid = np.isfinite(scores) & (scores > 0)
        result.append((candidates[valid], scores[valid]))

    return result


def _select_jump_cutpoints(
    candidates: list[JumpCandidates],
    quotas: NDArray[np.integer],
) -> list[NDArray[np.float64]]:
    cutpoints: list[NDArray[np.float64]] = []
    for (values, scores), quota in zip(candidates, quotas, strict=True):
        selected = np.argsort(-scores, kind='stable')[: int(quota)]
        cutpoints.append(np.sort(values[selected]))
    return cutpoints


def shap_jump_cutpoints(
    X: NDArray,
    shap_values: NDArray,
    *,
    n_thresholds: int,
) -> list[NDArray[np.float64]]:
    candidates = _shap_jump_candidates(X, shap_values)
    quotas = np.full(len(candidates), n_thresholds, dtype=np.int64)
    return _select_jump_cutpoints(candidates, quotas)


def _allocate_thresholds(
    attention: NDArray,
    capacities: NDArray,
    budget: int,
) -> NDArray[np.int64]:
    attention = np.asarray(attention, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    quotas = np.zeros(len(attention), dtype=np.int64)
    heap: list[tuple[float, int]] = []
    for column in np.flatnonzero(capacities):
        heappush(heap, (-attention[column], int(column)))

    for _ in range(min(budget, int(capacities.sum()))):
        _priority, column = heappop(heap)
        quotas[column] += 1
        if quotas[column] < capacities[column]:
            heappush(
                heap,
                (-attention[column] / (quotas[column] + 1), column),
            )
    return quotas


def _degroup_attention(token_attention: NDArray) -> NDArray[np.float64]:
    token_attention = np.asarray(token_attention, dtype=np.float64)
    feature_attention = np.zeros_like(token_attention)
    token_indices = np.arange(len(token_attention))
    for shift in FEATURE_GROUP_SHIFTS:
        np.add.at(
            feature_attention,
            (token_indices + shift) % len(token_attention),
            token_attention / len(FEATURE_GROUP_SHIFTS),
        )
    return feature_attention


def tabpfn_feature_attention(classifier, X: NDArray) -> NDArray[np.float64]:
    aggregator = classifier.models_[0].column_aggregator
    attention = aggregator.blocks[-1].attention
    pending_queries: list[torch.Tensor] = []
    row_scores: list[torch.Tensor] = []

    def capture_query(_module, _inputs, output):
        pending_queries.append(output.detach())

    def capture_keys(_module, _inputs, output):
        query = pending_queries.pop()
        key = output.detach()
        query = query.reshape(
            len(query),
            aggregator.num_cls_tokens,
            attention.num_heads,
            attention.head_dim,
        )
        key = key.reshape(
            len(key),
            key.shape[1],
            attention.num_heads,
            attention.head_dim,
        )
        query = aggregator.rope.rotate_queries_or_keys(query.transpose(1, 2)).transpose(
            1, 2
        )
        key = aggregator.rope.rotate_queries_or_keys(key.transpose(1, 2)).transpose(
            1, 2
        )
        logits = torch.einsum('bqhd,bkhd->bhqk', query.float(), key.float())
        logits *= attention.head_dim**-0.5
        weights = logits[..., aggregator.num_cls_tokens :].softmax(dim=-1)
        row_scores.append(weights.mean(dim=(1, 2)).cpu())

    query_handle = attention.q_projection.register_forward_hook(capture_query)
    key_handle = attention.k_projection.register_forward_hook(capture_keys)
    try:
        classifier.predict_proba(X)
    finally:
        query_handle.remove()
        key_handle.remove()

    token_attention = torch.cat(row_scores)[-len(X) :].mean(dim=0).numpy()
    feature_attention = _degroup_attention(token_attention)
    features = classifier.executor_.ensemble_members[0].feature_schema.features
    original_attention = np.zeros(classifier.n_features_in_, dtype=np.float64)
    for feature, score in zip(features, feature_attention, strict=True):
        original_attention[int(feature.ancestor[1:])] = score
    return original_attention


def _quantile_candidates(X: NDArray) -> list[QuantileCandidates]:
    X = np.asarray(X, dtype=np.float64)
    result: list[QuantileCandidates] = []
    for column in range(X.shape[1]):
        values, counts = np.unique(
            X[np.isfinite(X[:, column]), column],
            return_counts=True,
        )
        if len(values) < 2:
            empty = np.empty(0, dtype=np.float64)
            result.append((empty, empty))
            continue
        cutpoints = values[:-1] + (values[1:] - values[:-1]) / 2
        cumulative_probability = np.cumsum(counts)[:-1] / counts.sum()
        result.append((cutpoints, cumulative_probability))
    return result


def _select_quantile_cutpoints(
    candidates: list[QuantileCandidates],
    quotas: NDArray[np.integer],
) -> list[NDArray[np.float64]]:
    result: list[NDArray[np.float64]] = []
    for (cutpoints, probabilities), quota in zip(candidates, quotas, strict=True):
        quota = int(quota)
        if quota == 0:
            result.append(np.empty(0, dtype=np.float64))
            continue

        targets = np.arange(1, quota + 1) / (quota + 1)
        selected = np.empty(quota, dtype=np.int64)
        lower = 0
        for index, target in enumerate(targets):
            remaining = quota - index - 1
            upper = len(cutpoints) - remaining
            selected[index] = lower + np.argmin(
                np.abs(probabilities[lower:upper] - target)
            )
            lower = selected[index] + 1
        result.append(cutpoints[selected])
    return result


def attention_quantile_cutpoints(
    X: NDArray,
    attention: NDArray,
    *,
    n_thresholds: int,
) -> list[NDArray[np.float64]]:
    candidates = _quantile_candidates(X)
    capacities = np.asarray([len(values) for values, _probability in candidates])
    quotas = _allocate_thresholds(
        attention,
        capacities,
        n_thresholds * X.shape[1],
    )
    return _select_quantile_cutpoints(candidates, quotas)


def tabpfn_shap_cutpoints(
    X: NDArray,
    y: NDArray,
    *,
    continuous_start: int,
    n_thresholds: int,
    model_path: Path,
    random_state: int,
) -> list[NDArray[np.float64]]:
    X = np.asarray(X)
    y = np.asarray(y)
    classifier = _make_classifier(model_path, random_state)
    classifier.fit(X, y)
    X_sample, shap_values = _explain_sample(classifier, X, y, random_state)
    return shap_jump_cutpoints(
        X_sample[:, continuous_start:],
        shap_values[:, continuous_start:],
        n_thresholds=n_thresholds,
    )


def tabpfn_attention_cutpoints(
    X: NDArray,
    y: NDArray,
    *,
    continuous_start: int,
    n_thresholds: int,
    model_path: Path,
    random_state: int,
) -> list[NDArray[np.float64]]:
    X = np.asarray(X)
    y = np.asarray(y)
    classifier = _make_attention_classifier(model_path, random_state)
    classifier.fit(X, y)
    X_sample = X[_sample_indices(y, random_state)]
    attention = tabpfn_feature_attention(classifier, X_sample)
    return attention_quantile_cutpoints(
        X[:, continuous_start:],
        attention[continuous_start:],
        n_thresholds=n_thresholds,
    )
