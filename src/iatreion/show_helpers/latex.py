import numpy as np
import pandas as pd

from iatreion.configs import ShowResultConfig

from .common import (
    LoadedResult,
    MetricDictGetter,
    MetricFormatter,
    MetricValues,
    ResultComparator,
    _compare_delong_auc,
    _compare_wilcoxon_auc,
    _format_ci,
    _format_delta_pp,
    _format_mean_std,
    _format_pvalue,
    _prepare_results,
    _resolve_metrics,
)


def _build_comparison_table(
    results: list[LoadedResult],
    ref: LoadedResult,
    *,
    metrics: list[str] | None,
    metric_dict_getter: MetricDictGetter,
    metric_formatter: MetricFormatter,
    missing_metric_values: MetricValues,
    pvalue_name: str,
    comparator: ResultComparator,
) -> pd.DataFrame:
    metric_dicts = [metric_dict_getter(result) for result in results]
    metric_names = _resolve_metrics(metrics, metric_dicts)
    delta_col = f'Delta AUC (pp) vs {ref.label}'
    pvalue_col = f'{pvalue_name} p vs {ref.label}'

    rows: list[dict[str, str]] = []
    for result, metric_dict in zip(results, metric_dicts, strict=True):
        row: dict[str, str] = {'Model': result.label}
        for metric in metric_names:
            row[metric] = metric_formatter(
                metric_dict.get(metric, missing_metric_values)
            )
        if result.label == ref.label:
            row[delta_col] = '-'
            row[pvalue_col] = '-'
        else:
            delta, pvalue = comparator(ref, result)
            row[delta_col] = _format_delta_pp(delta)
            row[pvalue_col] = _format_pvalue(pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def _format_mean_std_values(values: MetricValues) -> str:
    mean, std = values
    return _format_mean_std(mean, std)


def _format_ci_values(values: MetricValues) -> str:
    point, lower, upper = values
    return _format_ci(point, lower, upper)


def make_mean_std_wilcoxon_table(config: ShowResultConfig) -> pd.DataFrame:
    results, ref = _prepare_results(config)
    return _build_comparison_table(
        results,
        ref,
        metrics=config.metrics,
        metric_dict_getter=lambda result: result.mean_std,
        metric_formatter=_format_mean_std_values,
        missing_metric_values=(np.nan, np.nan),
        pvalue_name='Wilcoxon',
        comparator=_compare_wilcoxon_auc,
    )


def make_ci_delong_table(config: ShowResultConfig) -> pd.DataFrame:
    results, ref = _prepare_results(config)
    return _build_comparison_table(
        results,
        ref,
        metrics=config.metrics,
        metric_dict_getter=lambda result: result.ci,
        metric_formatter=_format_ci_values,
        missing_metric_values=(np.nan, np.nan, np.nan),
        pvalue_name='DeLong',
        comparator=_compare_delong_auc,
    )
