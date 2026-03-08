import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray

from iatreion.configs import ShowResultConfig

from .common import (
    LoadedResult,
    ResultComparator,
    _compare_delong_auc,
    _compare_wilcoxon_auc,
    _format_pvalue,
    _prepare_results,
)


def _build_pairwise_pvalue_matrix(
    results: list[LoadedResult], comparator: ResultComparator
) -> tuple[pd.DataFrame, NDArray[np.object_]]:
    labels = [result.label for result in results]
    size = len(results)
    values = np.empty((size, size), dtype=float)
    annot = np.empty((size, size), dtype=object)
    for i, left in enumerate(results):
        for j, right in enumerate(results):
            if i == j:
                pvalue = 1.0
                annot[i, j] = '-'
            else:
                _, pvalue = comparator(left, right)
                annot[i, j] = _format_pvalue(pvalue)
            values[i, j] = pvalue
    matrix = pd.DataFrame(values, index=labels, columns=labels)
    return matrix, annot


def _plot_pvalue_heatmap(
    matrix: pd.DataFrame, annot: NDArray[np.object_], title: str
) -> Figure:
    side = max(6.0, len(matrix) * 0.8 + 2.0)
    fig, ax = plt.subplots(figsize=(side, side), layout='constrained')
    sns.heatmap(
        matrix,
        cmap='Reds_r',
        vmin=0,
        vmax=1,
        annot=annot,
        fmt='',
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'p-value'},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Model')
    return fig


def wilcoxon_pvalue_heatmap(config: ShowResultConfig) -> tuple[pd.DataFrame, Figure]:
    results, _ = _prepare_results(config)
    matrix, annot = _build_pairwise_pvalue_matrix(results, _compare_wilcoxon_auc)
    fig = _plot_pvalue_heatmap(matrix, annot, config.title)
    return matrix, fig


def delong_pvalue_heatmap(config: ShowResultConfig) -> tuple[pd.DataFrame, Figure]:
    results, _ = _prepare_results(config)
    matrix, annot = _build_pairwise_pvalue_matrix(results, _compare_delong_auc)
    fig = _plot_pvalue_heatmap(matrix, annot, config.title)
    return matrix, fig
