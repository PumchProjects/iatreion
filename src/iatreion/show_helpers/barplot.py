import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from iatreion.configs import ShowPerformanceConfig

from .performance import (
    LoadedResult,
    MetricSummaryGetter,
    PairwisePvalueGetter,
    _delong_pvalue,
    _format_pvalue,
    _get_ci_metric_summary,
    _get_fold_metric_values,
    _get_mean_std_metric_summary,
    _mcnemar_pvalue,
    _prepare_results,
    _pvalue_to_stars,
    _wilcoxon_pvalue,
)


def _build_metric_significance_barplot(
    results: list[LoadedResult],
    ref: LoadedResult,
    *,
    metric: str,
    title: str,
    ylabel: str,
    summary_getter: MetricSummaryGetter,
    pvalue_getter: PairwisePvalueGetter,
    test_name: str,
) -> tuple[pd.DataFrame, Figure]:
    labels = [result.label for result in results]
    values: list[float] = []
    err_lows: list[float] = []
    err_highs: list[float] = []
    display_values: list[str] = []
    for result in results:
        value, err_low, err_high, display = summary_getter(result, metric)
        values.append(value)
        err_lows.append(err_low)
        err_highs.append(err_high)
        display_values.append(display)

    pvalues: list[float] = []
    for result in results:
        if result.label == ref.label:
            pvalues.append(np.nan)
            continue
        pvalues.append(pvalue_getter(ref, result))

    values_arr = np.asarray(values, dtype=float)
    err_low_arr = np.asarray(err_lows, dtype=float)
    err_high_arr = np.asarray(err_highs, dtype=float)
    yerr = np.nan_to_num(np.vstack([err_low_arr, err_high_arr]), nan=0.0)

    width = max(7.0, len(results) * 1.2 + 2.0)
    fig, ax = plt.subplots(figsize=(width, 6), layout='constrained')
    colors = ['#4e79a7' for _ in results]
    colors[labels.index(ref.label)] = '#f28e2b'
    bars = ax.bar(
        labels,
        values_arr,
        yerr=yerr,
        capsize=6,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
    )

    upper = values_arr + np.nan_to_num(err_high_arr, nan=0.0)
    finite_upper = upper[np.isfinite(upper)]
    top = finite_upper.max() if finite_upper.size else 1.0
    offset = max(0.01, top * 0.04)
    for idx, bar in enumerate(bars):
        stars = _pvalue_to_stars(pvalues[idx])
        if not stars or labels[idx] == ref.label:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            upper[idx] + offset,
            stars,
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
        )

    ax.set(
        title=f'{title}\nStars vs {ref.label}: * p<0.05, ** p<0.01',
        xlabel='Model',
        ylabel=ylabel,
    )
    ax.set_ylim(0, max(1.0, top + 4 * offset))
    ax.tick_params(axis='x', labelrotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha('right')
    ax.yaxis.set_major_formatter(lambda value, _: f'{value:.0%}')

    pvalue_col = f'{test_name} p vs {ref.label}'
    rows: list[dict[str, str]] = []
    for label, display, pvalue in zip(labels, display_values, pvalues, strict=True):
        if label == ref.label:
            pvalue_text = '-'
            stars = '-'
        else:
            pvalue_text = _format_pvalue(pvalue)
            stars = _pvalue_to_stars(pvalue)
        rows.append(
            {'Model': label, metric: display, pvalue_col: pvalue_text, 'Sig': stars}
        )
    return pd.DataFrame(rows), fig


def auc_delong_ci_barplot(config: ShowPerformanceConfig) -> tuple[pd.DataFrame, Figure]:
    results, ref = _prepare_results(config)
    return _build_metric_significance_barplot(
        results,
        ref,
        metric='AUC',
        title=config.title,
        ylabel='AUC',
        summary_getter=_get_ci_metric_summary,
        pvalue_getter=lambda left, right: _delong_pvalue(
            left.y_true, left.y_score_pos, right.y_score_pos
        ),
        test_name='DeLong',
    )


def acc_mcnemar_ci_barplot(
    config: ShowPerformanceConfig,
) -> tuple[pd.DataFrame, Figure]:
    results, ref = _prepare_results(config)
    return _build_metric_significance_barplot(
        results,
        ref,
        metric='ACC',
        title=config.title,
        ylabel='ACC',
        summary_getter=_get_ci_metric_summary,
        pvalue_getter=lambda left, right: _mcnemar_pvalue(
            left.y_true, left.y_pred, right.y_pred
        ),
        test_name='McNemar',
    )


def auc_wilcoxon_std_barplot(
    config: ShowPerformanceConfig,
) -> tuple[pd.DataFrame, Figure]:
    results, ref = _prepare_results(config)
    return _build_metric_significance_barplot(
        results,
        ref,
        metric='AUC',
        title=config.title,
        ylabel='AUC',
        summary_getter=_get_mean_std_metric_summary,
        pvalue_getter=lambda left, right: _wilcoxon_pvalue(
            _get_fold_metric_values(left, 'AUC'),
            _get_fold_metric_values(right, 'AUC'),
        ),
        test_name='Wilcoxon',
    )


def acc_wilcoxon_std_barplot(
    config: ShowPerformanceConfig,
) -> tuple[pd.DataFrame, Figure]:
    results, ref = _prepare_results(config)
    return _build_metric_significance_barplot(
        results,
        ref,
        metric='ACC',
        title=config.title,
        ylabel='ACC',
        summary_getter=_get_mean_std_metric_summary,
        pvalue_getter=lambda left, right: _wilcoxon_pvalue(
            _get_fold_metric_values(left, 'ACC'),
            _get_fold_metric_values(right, 'ACC'),
        ),
        test_name='Wilcoxon',
    )
