import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from numpy.typing import NDArray

from iatreion.configs import ImportanceMethod, ImportanceScope, ShowResultConfig
from iatreion.exceptions import IatreionException
from iatreion.models import ImportanceScore

plt.rcParams['font.family'] = 'DejaVu Sans, Noto Sans CJK JP'

_SCORE_PATTERN = re.compile(
    r'^score_(?P<method>native|permutation|shap)_'
    r'(?P<name>.+)_(?P<outer>\d+)_(?P<inner>\d+)\.json$'
)


@dataclass(frozen=True)
class ImportanceResult:
    label: str
    name: str
    folds: list[tuple[int, int]]
    values: NDArray[np.floating]
    summary: pd.DataFrame


def _parse_score_file(path: Path) -> tuple[ImportanceMethod, str, int, int] | None:
    match = _SCORE_PATTERN.match(path.name)
    if match is None:
        return None
    method = cast(ImportanceMethod, match.group('method'))
    name = match.group('name')
    outer = int(match.group('outer'))
    inner = int(match.group('inner'))
    return method, name, outer, inner


def _load_json_score(path: Path) -> ImportanceScore:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        raise IatreionException(
            'Failed to read importance file "$path": $error',
            path=str(path),
            error=str(error),
        ) from error
    except json.JSONDecodeError as error:
        raise IatreionException(
            'Invalid JSON in "$path": $error',
            path=str(path),
            error=str(error),
        ) from error
    if not isinstance(data, dict):
        raise IatreionException('Invalid importance object in "$path".', path=str(path))
    return {str(key): float(value) for key, value in data.items()}


def _select_scope(
    scores: dict[tuple[int, int], ImportanceScore], scope: ImportanceScope
) -> dict[tuple[int, int], ImportanceScore]:
    if scope == 'all':
        return scores
    selected: dict[int, tuple[int, tuple[int, int]]] = {}
    for key in sorted(scores):
        outer, inner = key
        if outer not in selected or inner > selected[outer][0]:
            selected[outer] = (inner, key)
    return {
        key: scores[key]
        for _, key in sorted(selected.values(), key=lambda item: item[1])
    }


def _to_matrix(
    scores: dict[tuple[int, int], ImportanceScore],
    *,
    use_abs: bool,
    normalize: bool,
) -> tuple[list[tuple[int, int]], list[str], NDArray[np.floating]]:
    if not scores:
        raise IatreionException('No importance scores were loaded.')

    folds = sorted(scores.keys())
    features = sorted({feature for score in scores.values() for feature in score})
    matrix = np.zeros((len(folds), len(features)), dtype=float)
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    for fold_idx, fold in enumerate(folds):
        for feature, value in scores[fold].items():
            matrix[fold_idx, feature_index[feature]] = value

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if use_abs:
        matrix = np.abs(matrix)
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sum,
            out=np.zeros_like(matrix),
            where=row_sum > 0,
        )
    return folds, features, matrix


def _summarize(
    label: str,
    name: str,
    folds: list[tuple[int, int]],
    features: list[str],
    matrix: NDArray[np.floating],
) -> pd.DataFrame:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    frame = pd.DataFrame(
        {
            'Model': label,
            'Result': name,
            'Feature': features,
            'Mean': mean,
            'Std': std,
            'Fold Count': len(folds),
        }
    ).sort_values('Mean', ascending=False, ignore_index=True)
    frame['Rank'] = np.arange(1, len(frame) + 1)
    return frame


def _load_importance_result(
    config: ShowResultConfig,
    *,
    log_dir: Path,
    name: str,
    label: str,
    method: ImportanceMethod,
) -> ImportanceResult:
    scores: dict[tuple[int, int], ImportanceScore] = {}
    for path in log_dir.glob('score_*.json'):
        parsed = _parse_score_file(path)
        if parsed is None:
            continue
        parsed_method, parsed_name, outer, inner = parsed
        if parsed_method != method or parsed_name != name:
            continue
        scores[(outer, inner)] = _load_json_score(path)

    scores = _select_scope(scores, config.importance_scope)
    if not scores:
        raise IatreionException(
            'No "$method" importance files found for "$name" in "$path".',
            method=method,
            name=name,
            path=str(log_dir),
        )
    folds, features, matrix = _to_matrix(
        scores,
        use_abs=config.importance_abs,
        normalize=config.importance_normalize,
    )
    summary = _summarize(label, name, folds, features, matrix)
    return ImportanceResult(label, name, folds, matrix, summary)


def _prepare_importance_results(config: ShowResultConfig) -> list[ImportanceResult]:
    results: list[ImportanceResult] = []
    seen_labels: set[str] = set()
    for train, name, label, method in config.make_configs():
        if label in seen_labels:
            raise IatreionException('Duplicate model label "$label".', label=label)
        seen_labels.add(label)
        results.append(
            _load_importance_result(
                config,
                log_dir=train._log_dir,
                name=name,
                label=label,
                method=method,
            )
        )
    return results


def feature_importance_barplot(config: ShowResultConfig) -> tuple[pd.DataFrame, Figure]:
    results = _prepare_importance_results(config)
    top_k = max(1, config.importance_top_k)
    n_results = len(results)
    fig_h = max(4.0, min(0.4 * top_k + 2.5, 14.0))
    fig_w = max(7.0, 6.0 * n_results)
    fig, axes = plt.subplots(
        1, n_results, figsize=(fig_w, fig_h), layout='constrained', squeeze=False
    )

    table_list: list[pd.DataFrame] = []
    for idx, result in enumerate(results):
        ax = axes[0, idx]
        top = result.summary.head(top_k).iloc[::-1]
        ax.barh(
            top['Feature'],
            top['Mean'],
            xerr=top['Std'],
            capsize=3,
            color='#4e79a7',
            edgecolor='black',
            linewidth=0.6,
        )
        if config.importance_normalize:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(
            f'{result.label} ({result.name})\n{len(result.folds)} folds',
            fontsize=10,
        )
        ax.set_xlabel('Importance')
        if idx == 0:
            ax.set_ylabel('Feature')
        table_list.append(top)

    fig.suptitle(config.title, fontsize=12)
    table = pd.concat(table_list, ignore_index=True)
    return table, fig


def feature_importance_heatmap(config: ShowResultConfig) -> tuple[pd.DataFrame, Figure]:
    results = _prepare_importance_results(config)
    top_k = max(1, config.importance_top_k)
    summary_map = {
        result.label: result.summary.set_index('Feature')['Mean'] for result in results
    }
    merged = pd.DataFrame(summary_map).fillna(0.0)
    global_rank = merged.mean(axis=1).sort_values(ascending=False)
    top_features = global_rank.head(top_k).index
    matrix = merged.loc[top_features]

    fig_h = max(4.0, min(0.45 * len(top_features) + 2.5, 14.0))
    fig_w = max(6.5, 1.6 * len(results) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
    sns.heatmap(
        matrix,
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Importance'},
        ax=ax,
    )
    ax.set_xlabel('Model')
    ax.set_ylabel('Feature')
    ax.set_title(config.title)
    return matrix.reset_index(names='Feature'), fig
