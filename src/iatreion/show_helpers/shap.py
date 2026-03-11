import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from iatreion.configs import ShowShapConfig
from iatreion.exceptions import IatreionException

from .importance import FoldKey, _select_scope

_SHAP_PATTERN = re.compile(r'^shap_(?P<name>.+)_(?P<outer>\d+)_(?P<inner>\d+)\.npz$')


@dataclass(frozen=True)
class ShapResult:
    label: str
    name: str
    folds: list[FoldKey]
    explanation: shap.Explanation
    outer_folds: NDArray[np.integer]
    inner_folds: NDArray[np.integer]
    sample_indices: NDArray[np.integer]
    y_true: NDArray[np.integer]


def _parse_shap_file(path: Path) -> tuple[str, int, int] | None:
    match = _SHAP_PATTERN.match(path.name)
    if match is None:
        return None
    name = match.group('name')
    outer = int(match.group('outer'))
    inner = int(match.group('inner'))
    return name, outer, inner


def _load_shap_explanation(
    path: Path,
) -> tuple[shap.Explanation, NDArray[np.integer], NDArray[np.integer]]:
    try:
        with np.load(path) as arrays:
            required = {
                'values',
                'base_values',
                'data',
                'y_true',
                'sample_indices',
                'feature_names',
            }
            missing = required.difference(arrays.files)
            if missing:
                raise IatreionException(
                    'Missing SHAP keys $keys in "$path".',
                    keys=', '.join(sorted(missing)),
                    path=str(path),
                )
            values = np.asarray(arrays['values'], dtype=float)
            base_values = np.asarray(arrays['base_values'], dtype=float)
            data = np.asarray(arrays['data'], dtype=float)
            y_true = np.asarray(arrays['y_true'], dtype=int).reshape(-1)
            sample_indices = np.asarray(arrays['sample_indices'], dtype=int).reshape(-1)
            feature_names = [str(name) for name in arrays['feature_names'].tolist()]
            output_names = (
                [str(name) for name in arrays['output_names'].tolist()]
                if 'output_names' in arrays.files
                else None
            )
    except OSError as error:
        raise IatreionException(
            'Failed to read SHAP file "$path": $error',
            path=str(path),
            error=str(error),
        ) from error

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
        output_names=output_names,
    )
    return explanation, sample_indices, y_true


def _concat_shap_explanations(explanations: list[shap.Explanation]) -> shap.Explanation:
    first = explanations[0]
    return shap.Explanation(
        values=np.concatenate(
            [
                np.asarray(explanation.values, dtype=float)
                for explanation in explanations
            ],
            axis=0,
        ),
        base_values=np.concatenate(
            [
                np.asarray(explanation.base_values, dtype=float)
                for explanation in explanations
            ],
            axis=0,
        ),
        data=np.concatenate(
            [np.asarray(explanation.data, dtype=float) for explanation in explanations],
            axis=0,
        ),
        feature_names=list(first.feature_names),
        output_names=(
            list(first.output_names) if first.output_names is not None else None
        ),
    )


def _load_shap_result(
    config: ShowShapConfig,
    *,
    log_dir: Path,
    name: str,
    label: str,
) -> ShapResult:
    paths: dict[FoldKey, Path] = {}
    for path in log_dir.glob('shap_*.npz'):
        parsed = _parse_shap_file(path)
        if parsed is None:
            continue
        parsed_name, outer, inner = parsed
        if parsed_name != name:
            continue
        paths[(outer, inner)] = path

    paths = _select_scope(paths, config.fold_scope)
    if not paths:
        raise IatreionException(
            'No SHAP artifact files found for "$name" in "$path".',
            name=name,
            path=str(log_dir),
        )

    explanations: list[shap.Explanation] = []
    outer_folds: list[NDArray[np.integer]] = []
    inner_folds: list[NDArray[np.integer]] = []
    sample_indices: list[NDArray[np.integer]] = []
    y_true: list[NDArray[np.integer]] = []
    folds = sorted(paths.keys())
    for outer, inner in folds:
        explanation, indices, y_fold = _load_shap_explanation(paths[(outer, inner)])
        explanations.append(explanation)
        outer_folds.append(np.full(indices.shape[0], outer, dtype=np.int64))
        inner_folds.append(np.full(indices.shape[0], inner, dtype=np.int64))
        sample_indices.append(indices)
        y_true.append(y_fold)

    return ShapResult(
        label=label,
        name=name,
        folds=folds,
        explanation=_concat_shap_explanations(explanations),
        outer_folds=np.concatenate(outer_folds, axis=0),
        inner_folds=np.concatenate(inner_folds, axis=0),
        sample_indices=np.concatenate(sample_indices, axis=0),
        y_true=np.concatenate(y_true, axis=0),
    )


def _prepare_shap_results(config: ShowShapConfig) -> list[ShapResult]:
    results: list[ShapResult] = []
    seen_labels: set[str] = set()
    for train, name, label in config.make_config():
        if label in seen_labels:
            raise IatreionException('Duplicate model label "$label".', label=label)
        seen_labels.add(label)
        results.append(
            _load_shap_result(
                config,
                log_dir=train._log_dir,
                name=name,
                label=label,
            )
        )
    return results


def _require_single_shap_result(
    results: list[ShapResult],
    plot_name: str,
) -> ShapResult:
    if len(results) != 1:
        raise IatreionException(
            '"$plot" currently requires exactly one model/result.',
            plot=plot_name,
        )
    return results[0]


def _resolve_shap_output_index(
    explanation: shap.Explanation,
    requested: int | None,
) -> int:
    values = np.asarray(explanation.values)
    n_outputs = 1 if values.ndim == 2 else values.shape[-1]
    if requested is not None:
        if requested < 0 or requested >= n_outputs:
            raise IatreionException(
                'Invalid SHAP output index $index; expected 0 <= index < $n.',
                index=str(requested),
                n=str(n_outputs),
            )
        return requested
    if n_outputs == 1:
        return 0
    if n_outputs == 2:
        return 1
    raise IatreionException(
        'SHAP plot found $n outputs; please set --shap-output-index/-soi.',
        n=str(n_outputs),
    )


def _get_output_label(explanation: shap.Explanation, output_index: int) -> str:
    output_names = explanation.output_names
    if output_names is None:
        return f'output_{output_index}'
    if isinstance(output_names, str):
        return output_names
    names = list(output_names)
    if len(names) == 1:
        return names[0]
    return names[output_index]


def _select_shap_output(
    explanation: shap.Explanation,
    output_index: int,
) -> shap.Explanation:
    values = np.asarray(explanation.values, dtype=float)
    data = np.asarray(explanation.data, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)
    feature_names = list(explanation.feature_names)
    output_label = _get_output_label(explanation, output_index)

    if values.ndim == 2:
        selected_values = values
        selected_base_values = base_values
    else:
        selected_values = values[:, :, output_index]
        if base_values.ndim == 1:
            selected_base_values = base_values
        else:
            selected_base_values = base_values[:, output_index]

    return shap.Explanation(
        values=selected_values,
        base_values=selected_base_values,
        data=data,
        feature_names=feature_names,
        output_names=output_label,
    )


def _summarize_shap(
    result: ShapResult,
    explanation: shap.Explanation,
    output_label: str,
) -> pd.DataFrame:
    values = np.abs(np.asarray(explanation.values, dtype=float))
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    frame = pd.DataFrame(
        {
            'Model': result.label,
            'Result': result.name,
            'Output': output_label,
            'Feature': list(explanation.feature_names),
            'Mean |SHAP|': mean,
            'Std |SHAP|': std,
            'Sample Count': explanation.shape[0],
            'Fold Count': len(result.folds),
        }
    ).sort_values('Mean |SHAP|', ascending=False, ignore_index=True)
    frame['Rank'] = np.arange(1, len(frame) + 1)
    return frame


def _get_feature_index(explanation: shap.Explanation, feature_name: str) -> int:
    feature_names = list(explanation.feature_names)
    if feature_name not in feature_names:
        raise IatreionException(
            'Unknown feature "$feature". Available feature count: $n.',
            feature=feature_name,
            n=str(len(feature_names)),
        )
    return feature_names.index(feature_name)


def _select_sample(result: ShapResult, sample_index: int) -> None:
    n_samples = result.sample_indices.shape[0]
    if sample_index < 0 or sample_index >= n_samples:
        raise IatreionException(
            'Invalid SHAP sample index $index; expected 0 <= index < $n.',
            index=str(sample_index),
            n=str(n_samples),
        )


def _select_shap_sample(
    explanation: shap.Explanation,
    sample_index: int,
) -> shap.Explanation:
    return explanation[sample_index]


def shap_summary_plot(config: ShowShapConfig) -> tuple[pd.DataFrame, Figure]:
    results = _prepare_shap_results(config)
    top_k = max(1, config.top_k)
    n_results = len(results)
    fig_h = max(4.5, min(0.35 * top_k + 3.0, 14.0))
    fig_w = max(7.0, 6.5 * n_results)
    fig, axes = plt.subplots(
        1, n_results, figsize=(fig_w, fig_h), layout='constrained', squeeze=False
    )

    table_list: list[pd.DataFrame] = []
    for idx, result in enumerate(results):
        output_index = _resolve_shap_output_index(
            result.explanation,
            config.shap_output_index,
        )
        output_label = _get_output_label(result.explanation, output_index)
        explanation = _select_shap_output(result.explanation, output_index)
        ax = axes[0, idx]
        shap.plots.beeswarm(
            explanation,
            max_display=top_k,
            ax=ax,
            show=False,
            plot_size=None,
            color_bar=idx == n_results - 1,
        )
        ax.set_title(
            f'{result.label} ({result.name})\n{output_label}, '
            f'{explanation.shape[0]} samples',
            fontsize=10,
        )
        if idx != 0:
            ax.set_ylabel('')
        table_list.append(
            _summarize_shap(result, explanation, output_label).head(top_k)
        )

    fig.suptitle(config.title, fontsize=12)
    return pd.concat(table_list, ignore_index=True), fig


def shap_waterfall_plot(config: ShowShapConfig) -> tuple[pd.DataFrame, Figure]:
    result = _require_single_shap_result(
        _prepare_shap_results(config),
        'shap waterfall',
    )
    _select_sample(result, config.shap_sample_index)
    output_index = _resolve_shap_output_index(
        result.explanation,
        config.shap_output_index,
    )
    output_label = _get_output_label(result.explanation, output_index)
    explanation = _select_shap_output(result.explanation, output_index)

    plt.figure(
        figsize=(9.0, max(4.5, 0.35 * config.top_k + 3.0)),
        layout='constrained',
    )
    sample_exp = _select_shap_sample(explanation, config.shap_sample_index)
    shap.plots.waterfall(
        sample_exp,
        max_display=max(1, config.top_k),
        show=False,
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(
        f'{config.title or result.label} ({output_label})\n'
        f'outer={result.outer_folds[config.shap_sample_index]}, '
        f'inner={result.inner_folds[config.shap_sample_index]}, '
        f'sample={result.sample_indices[config.shap_sample_index]}'
    )

    values = np.asarray(sample_exp.values, dtype=float)
    data = np.asarray(sample_exp.data, dtype=float)
    feature_names = list(explanation.feature_names)
    table = pd.DataFrame(
        {
            'Model': result.label,
            'Result': result.name,
            'Output': output_label,
            'Outer Fold': result.outer_folds[config.shap_sample_index],
            'Inner Fold': result.inner_folds[config.shap_sample_index],
            'Fold Sample Index': result.sample_indices[config.shap_sample_index],
            'y_true': result.y_true[config.shap_sample_index],
            'Base Value': float(np.asarray(sample_exp.base_values).reshape(-1)[0]),
            'Prediction': float(
                np.asarray(sample_exp.base_values).reshape(-1)[0] + values.sum()
            ),
            'Feature': feature_names,
            'Feature Value': data,
            'SHAP Value': values,
            'Abs SHAP': np.abs(values),
        }
    ).sort_values('Abs SHAP', ascending=False, ignore_index=True)
    table['Rank'] = np.arange(1, len(table) + 1)
    return table, fig


def shap_dependence_plot(config: ShowShapConfig) -> tuple[pd.DataFrame, Figure]:
    if config.shap_feature is None:
        raise IatreionException('SHAP dependence plot requires --shap-feature/-sf.')

    result = _require_single_shap_result(
        _prepare_shap_results(config),
        'shap dependence',
    )
    output_index = _resolve_shap_output_index(
        result.explanation,
        config.shap_output_index,
    )
    output_label = _get_output_label(result.explanation, output_index)
    explanation = _select_shap_output(result.explanation, output_index)
    feature_index = _get_feature_index(explanation, config.shap_feature)

    color_source: shap.Explanation | str | None
    if config.shap_color_feature is None:
        color_source = explanation
    else:
        color_index = _get_feature_index(explanation, config.shap_color_feature)
        color_source = explanation[:, color_index]

    fig, ax = plt.subplots(figsize=(8.5, 6.0), layout='constrained')
    shap.plots.scatter(
        explanation[:, feature_index],
        color=color_source,
        ax=ax,
        show=False,
        title=config.title or None,
    )
    ax.set_title(
        config.title or f'{result.label} ({output_label}) on {config.shap_feature}'
    )

    values = np.asarray(explanation.values, dtype=float)
    data = np.asarray(explanation.data, dtype=float)
    table_dict: dict[str, object] = {
        'Model': result.label,
        'Result': result.name,
        'Output': output_label,
        'Outer Fold': result.outer_folds,
        'Inner Fold': result.inner_folds,
        'Fold Sample Index': result.sample_indices,
        'y_true': result.y_true,
        'Feature': config.shap_feature,
        'Feature Value': data[:, feature_index],
        'SHAP Value': values[:, feature_index],
    }
    if config.shap_color_feature is not None:
        color_index = _get_feature_index(explanation, config.shap_color_feature)
        table_dict['Color Feature'] = config.shap_color_feature
        table_dict['Color Feature Value'] = data[:, color_index]

    table = pd.DataFrame(table_dict).sort_values(
        'Feature Value',
        ignore_index=True,
    )
    return table, fig
