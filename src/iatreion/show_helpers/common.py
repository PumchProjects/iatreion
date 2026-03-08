import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from iatreion.configs import ShowResultConfig
from iatreion.exceptions import IatreionException

type MeanStdMetrics = dict[str, tuple[float, float]]
type CiMetrics = dict[str, tuple[float, float, float]]
type MetricValues = tuple[float, ...]
type MetricDict = dict[str, MetricValues]
type MetricDictGetter = Callable[['LoadedResult'], MetricDict]
type MetricFormatter = Callable[[MetricValues], str]
type ResultComparator = Callable[['LoadedResult', 'LoadedResult'], tuple[float, float]]
type ResultValidator = Callable[[list['LoadedResult']], None]
type PairwisePvalueGetter = Callable[['LoadedResult', 'LoadedResult'], float]
type MetricSummaryGetter = Callable[
    ['LoadedResult', str], tuple[float, float, float, str]
]

DEFAULT_BINARY_METRICS: list[str] = ['AUC', 'ACC', 'P', 'R', 'F1', 'SEN', 'SPC']
RATE_PATTERN = r'(?:nan|[+-]?\d+(?:\.\d+)?)'
MEAN_STD_PATTERN = re.compile(
    rf'^(?P<metric>[A-Z0-9]+)\s+(?P<mean>{RATE_PATTERN})%\s+\S+\s+'
    rf'(?P<std>{RATE_PATTERN})%$'
)
CI_PATTERN = re.compile(
    rf'^(?P<metric>[A-Z0-9]+)\s+(?P<point>{RATE_PATTERN})%\s+\['
    rf'(?P<lower>{RATE_PATTERN})%,\s+(?P<upper>{RATE_PATTERN})%\]$'
)


@dataclass(frozen=True)
class ResultSource:
    label: str
    name: str
    result_file: Path
    mean_std_file: Path
    ci_file: Path


@dataclass(frozen=True)
class LoadedResult:
    label: str
    name: str
    mean_std: MeanStdMetrics
    ci: CiMetrics
    fold_metrics: dict[str, NDArray[np.floating]]
    auc_folds: NDArray[np.floating]
    y_true: NDArray[np.integer]
    y_pred: NDArray[np.integer]
    y_score_pos: NDArray[np.floating]
    full_auc: float


def _parse_rate(value: str) -> float:
    if value.lower() == 'nan':
        return np.nan
    return float(value) / 100


def _format_rate(value: float) -> str:
    return 'nan' if np.isnan(value) else f'{value:.2%}'


def _format_mean_std(mean: float, std: float) -> str:
    return f'{_format_rate(mean)} +/- {_format_rate(std)}'


def _format_ci(point: float, lower: float, upper: float) -> str:
    return f'{_format_rate(point)} [{_format_rate(lower)}, {_format_rate(upper)}]'


def _format_delta_pp(value: float) -> str:
    return 'nan' if np.isnan(value) else f'{value * 100:+.2f} pp'


def _format_pvalue(value: float) -> str:
    if np.isnan(value):
        return 'nan'
    if value < 1e-4:
        return '<1e-4'
    return f'{value:.4f}'


def _parse_metric_lines(
    path: Path, pattern: re.Pattern[str], n_values: int
) -> dict[str, tuple[float, ...]]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise IatreionException(
            'Failed to read metric log "$path": $error',
            path=str(path),
            error=str(error),
        ) from error

    result: dict[str, tuple[float, ...]] = {}
    for line in lines:
        matched = pattern.match(line.strip())
        if matched is None:
            continue
        metric = matched.group('metric')
        values = tuple(
            _parse_rate(matched.group(name))
            for name in ('mean', 'std')
            if name in matched.groupdict()
        )
        if not values:
            values = tuple(
                _parse_rate(matched.group(name))
                for name in ('point', 'lower', 'upper')
                if name in matched.groupdict()
            )
        if len(values) != n_values:
            continue
        result[metric] = values
    if not result:
        raise IatreionException(
            'No metric lines were parsed from "$path".', path=str(path)
        )
    return result


def parse_mean_std_log(path: Path) -> MeanStdMetrics:
    parsed = _parse_metric_lines(path, MEAN_STD_PATTERN, n_values=2)
    return {key: (values[0], values[1]) for key, values in parsed.items()}


def parse_ci_log(path: Path) -> CiMetrics:
    parsed = _parse_metric_lines(path, CI_PATTERN, n_values=3)
    return {key: (values[0], values[1], values[2]) for key, values in parsed.items()}


def parse_result_sources(config: ShowResultConfig) -> list[ResultSource]:
    sources: list[ResultSource] = []
    labels: set[str] = set()
    for train, name, label in config.make_configs():
        result_file = train.get_results_file(name)
        if not result_file.is_file():
            raise IatreionException(
                'Result file "$path" does not exist.', path=str(result_file)
            )
        labels.add(label)
        mean_std_file = train.get_avg_log_file(name)
        ci_file = train.get_ci_log_file(name)
        if not mean_std_file.is_file():
            raise IatreionException(
                'Missing mean/std log "$path".', path=str(mean_std_file)
            )
        if not ci_file.is_file():
            raise IatreionException('Missing CI log "$path".', path=str(ci_file))
        sources.append(
            ResultSource(
                label=label,
                name=name,
                result_file=result_file,
                mean_std_file=mean_std_file,
                ci_file=ci_file,
            )
        )
    return sources


def _extract_pos_score(y_score: NDArray[np.floating]) -> NDArray[np.floating]:
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim != 2:
        raise IatreionException(
            'y_score should be 1D or 2D, got $ndim.', ndim=str(y_score.ndim)
        )
    if y_score.shape[1] == 1:
        return y_score[:, 0]
    return y_score[:, 1]


def _predict_labels(y_score: NDArray[np.floating]) -> NDArray[np.integer]:
    if y_score.ndim == 1:
        return (y_score >= 0.5).astype(int)
    if y_score.ndim != 2:
        raise IatreionException(
            'y_score should be 1D or 2D, got $ndim.', ndim=str(y_score.ndim)
        )
    if y_score.shape[1] == 1:
        return (y_score[:, 0] >= 0.5).astype(int)
    return np.argmax(y_score, axis=1).astype(int)


def _to_binary_target(y_true: NDArray[np.integer]) -> NDArray[np.integer]:
    labels = np.unique(y_true)
    if labels.size != 2:
        raise IatreionException(
            'AUC comparison currently supports binary tasks only; got $n classes.',
            n=str(labels.size),
        )
    pos_label = labels[-1]
    return (y_true == pos_label).astype(int)


def _safe_auc(y_true: NDArray[np.integer], y_score: NDArray[np.floating]) -> float:
    try:
        return float(roc_auc_score(_to_binary_target(y_true), y_score))
    except ValueError:
        return np.nan


def _load_npz(path: Path) -> dict[str, NDArray]:
    try:
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except OSError as error:
        raise IatreionException(
            'Failed to read result npz "$path": $error',
            path=str(path),
            error=str(error),
        ) from error


def _load_result(source: ResultSource) -> LoadedResult:
    arrays = _load_npz(source.result_file)
    try:
        y_true = np.asarray(arrays['y_true']).astype(int).reshape(-1)
        y_score = np.asarray(arrays['y_score']).astype(float)
    except KeyError as error:
        raise IatreionException(
            'Missing key "$key" in "$path".',
            key=error.args[0],
            path=str(source.result_file),
        ) from error

    fold_metrics = {
        key: np.asarray(value).astype(float).reshape(-1)
        for key, value in arrays.items()
        if key not in {'y_true', 'y_score'} and np.asarray(value).ndim == 1
    }
    if 'AUC' not in fold_metrics:
        raise IatreionException(
            'Missing key "$key" in "$path".', key='AUC', path=str(source.result_file)
        )
    auc_folds = fold_metrics['AUC']
    y_score_pos = _extract_pos_score(y_score)
    y_pred = _predict_labels(y_score)
    if y_true.shape[0] != y_score_pos.shape[0]:
        raise IatreionException(
            'Mismatched sample size in "$path": y_true=$n_true, y_score=$n_score.',
            path=str(source.result_file),
            n_true=str(y_true.shape[0]),
            n_score=str(y_score_pos.shape[0]),
        )

    mean_std = parse_mean_std_log(source.mean_std_file)
    ci = parse_ci_log(source.ci_file)
    return LoadedResult(
        label=source.label,
        name=source.name,
        mean_std=mean_std,
        ci=ci,
        fold_metrics=fold_metrics,
        auc_folds=auc_folds,
        y_true=y_true,
        y_pred=y_pred,
        y_score_pos=y_score_pos,
        full_auc=_safe_auc(y_true, y_score_pos),
    )


def _select_reference(
    results: list[LoadedResult], reference: str | None
) -> LoadedResult:
    if reference is None:
        return results[0]
    for result in results:
        if result.label == reference:
            return result
    raise IatreionException(
        'Reference model "$name" not found among labels: $labels.',
        name=reference,
        labels=', '.join(result.label for result in results),
    )


def _resolve_metrics(
    metrics: list[str] | None, metric_dicts: list[dict[str, tuple[float, ...]]]
) -> list[str]:
    if metrics:
        return metrics
    merged = {key for metric_dict in metric_dicts for key in metric_dict}
    ordered = [metric for metric in DEFAULT_BINARY_METRICS if metric in merged]
    extras = sorted(merged.difference(ordered))
    return ordered + extras


def _wilcoxon_pvalue(
    reference: NDArray[np.floating], target: NDArray[np.floating]
) -> float:
    mask = (~np.isnan(reference)) & (~np.isnan(target))
    ref = reference[mask]
    tar = target[mask]
    if ref.size == 0:
        return np.nan
    if np.allclose(ref, tar):
        return 1.0
    try:
        return float(stats.wilcoxon(ref, tar, method='auto').pvalue)
    except ValueError:
        return 1.0


def _compute_midrank(values: NDArray[np.floating]) -> NDArray[np.floating]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_ranks = np.empty(sorted_values.shape[0], dtype=float)
    i = 0
    while i < sorted_values.shape[0]:
        j = i
        while j < sorted_values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        sorted_ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _fast_delong(
    predictions: NDArray[np.floating], positive_count: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    total_count = predictions.shape[1]
    negative_count = total_count - positive_count
    pos = predictions[:, :positive_count]
    neg = predictions[:, positive_count:]

    tx = np.empty((predictions.shape[0], positive_count), dtype=float)
    ty = np.empty((predictions.shape[0], negative_count), dtype=float)
    tz = np.empty((predictions.shape[0], total_count), dtype=float)
    for i in range(predictions.shape[0]):
        tx[i] = _compute_midrank(pos[i])
        ty[i] = _compute_midrank(neg[i])
        tz[i] = _compute_midrank(predictions[i])

    aucs = tz[:, :positive_count].sum(axis=1)
    aucs = aucs / (positive_count * negative_count)
    aucs -= (positive_count + 1) / (2 * negative_count)

    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    covariance = sx / positive_count + sy / negative_count
    return aucs, covariance


def _delong_pvalue(
    y_true: NDArray[np.integer],
    reference_score: NDArray[np.floating],
    target_score: NDArray[np.floating],
) -> float:
    mask = (~np.isnan(reference_score)) & (~np.isnan(target_score))
    target = _to_binary_target(y_true[mask])
    if target.size == 0:
        return np.nan

    positive = target == 1
    positive_count = int(np.sum(positive))
    negative_count = target.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return np.nan

    order = np.r_[np.flatnonzero(positive), np.flatnonzero(~positive)]
    predictions = np.vstack([reference_score[mask], target_score[mask]])[:, order]
    aucs, covariance = _fast_delong(predictions, positive_count)
    diff = aucs[1] - aucs[0]
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if variance <= 0:
        return 1.0 if np.isclose(diff, 0) else 0.0
    z_value = np.abs(diff) / np.sqrt(variance)
    return float(2 * stats.norm.sf(z_value))


def _validate_y_true_consistency(results: list[LoadedResult]) -> None:
    baseline = results[0]
    for result in results[1:]:
        if np.array_equal(result.y_true, baseline.y_true):
            continue
        raise IatreionException(
            'y_true mismatch between "$left" and "$right". '
            'DeLong comparison requires the same full test labels.',
            left=baseline.label,
            right=result.label,
        )


def _prepare_results(
    config: ShowResultConfig,
) -> tuple[list[LoadedResult], LoadedResult]:
    sources = parse_result_sources(config)
    results = [_load_result(source) for source in sources]
    _validate_y_true_consistency(results)
    ref = _select_reference(results, config.reference)
    return results, ref


def _compare_wilcoxon_auc(
    reference: LoadedResult, target: LoadedResult
) -> tuple[float, float]:
    delta = np.nanmean(target.auc_folds) - np.nanmean(reference.auc_folds)
    pvalue = _wilcoxon_pvalue(reference.auc_folds, target.auc_folds)
    return delta, pvalue


def _compare_delong_auc(
    reference: LoadedResult, target: LoadedResult
) -> tuple[float, float]:
    delta = target.full_auc - reference.full_auc
    pvalue = _delong_pvalue(reference.y_true, reference.y_score_pos, target.y_score_pos)
    return delta, pvalue


def _get_fold_metric_values(result: LoadedResult, metric: str) -> NDArray[np.floating]:
    values = result.fold_metrics.get(metric)
    if values is None:
        raise IatreionException(
            'Metric "$metric" is missing in "$model".',
            metric=metric,
            model=result.label,
        )
    return values


def _mcnemar_pvalue(
    y_true: NDArray[np.integer],
    reference_pred: NDArray[np.integer],
    target_pred: NDArray[np.integer],
) -> float:
    ref_correct = reference_pred == y_true
    tar_correct = target_pred == y_true
    b = int(np.sum(ref_correct & ~tar_correct))
    c = int(np.sum(~ref_correct & tar_correct))
    if b + c == 0:
        return 1.0
    a = int(np.sum(ref_correct & tar_correct))
    d = int(np.sum(~ref_correct & ~tar_correct))
    table = np.array([[a, b], [c, d]])
    return float(mcnemar(table, exact=True).pvalue)


def _pvalue_to_stars(pvalue: float) -> str:
    if np.isnan(pvalue):
        return ''
    if pvalue < 0.01:
        return '**'
    if pvalue < 0.05:
        return '*'
    return 'ns'


def _get_ci_metric_summary(
    result: LoadedResult, metric: str
) -> tuple[float, float, float, str]:
    point, lower, upper = result.ci.get(metric, (np.nan, np.nan, np.nan))
    err_low = point - lower if not np.isnan(point) and not np.isnan(lower) else np.nan
    err_high = upper - point if not np.isnan(point) and not np.isnan(upper) else np.nan
    return point, err_low, err_high, _format_ci(point, lower, upper)


def _get_mean_std_metric_summary(
    result: LoadedResult, metric: str
) -> tuple[float, float, float, str]:
    mean, std = result.mean_std.get(metric, (np.nan, np.nan))
    return mean, std, std, _format_mean_std(mean, std)
