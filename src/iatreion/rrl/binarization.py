from pathlib import Path

import numpy as np
import shap
from numpy.typing import NDArray
from sklearn.utils import resample

MAX_SHAP_SAMPLES = 256


def _make_classifier(model_path: Path, random_state: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        model_path=str(model_path),
        n_estimators=1,
        auto_scale_n_estimators=False,
        random_state=random_state,
        n_preprocessing_jobs=1,
    )


def _sample_indices(y: NDArray, random_state: int) -> NDArray[np.integer]:
    indices = np.arange(len(y))
    if len(indices) <= MAX_SHAP_SAMPLES:
        return indices
    return np.sort(
        resample(
            indices,
            replace=False,
            n_samples=MAX_SHAP_SAMPLES,
            stratify=y,
            random_state=random_state,
        )
    )


def shap_jump_cutpoints(
    X: NDArray,
    shap_values: NDArray,
    *,
    n_thresholds: int,
) -> list[NDArray[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    values = np.asarray(shap_values, dtype=np.float64)
    values = values.reshape(len(X), X.shape[1], -1)
    cutpoints: list[NDArray[np.float64]] = []

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
            cutpoints.append(np.empty(0, dtype=np.float64))
            continue

        means = np.zeros((len(unique), feature_values.shape[1]), dtype=np.float64)
        np.add.at(means, inverse, feature_values)
        means /= counts[:, None]
        scores = np.linalg.norm(np.diff(means, axis=0), axis=1)
        candidates = unique[:-1] + (unique[1:] - unique[:-1]) / 2
        valid = np.isfinite(scores) & (scores > 0)
        candidates = candidates[valid]
        scores = scores[valid]

        selected = np.argsort(-scores, kind='stable')[:n_thresholds]
        cutpoints.append(np.sort(candidates[selected]))

    return cutpoints


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

    sample_indices = _sample_indices(y, random_state)
    X_sample = X[sample_indices]
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
    return shap_jump_cutpoints(
        X_sample[:, continuous_start:],
        np.asarray(explanation.values)[:, continuous_start:],
        n_thresholds=n_thresholds,
    )
