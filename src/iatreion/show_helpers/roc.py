import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from iatreion.configs import ShowPerformanceConfig

from .performance import (
    _delong_pvalue,
    _format_pvalue,
    _format_rate,
    _prepare_results,
    _to_binary_target,
)


def roc_delong_comparison_plot(
    config: ShowPerformanceConfig,
) -> tuple[pd.DataFrame, Figure]:
    results, _ = _prepare_results(config)
    best = max(
        results,
        key=lambda result: -np.inf if np.isnan(result.full_auc) else result.full_auc,
    )
    if np.isnan(best.full_auc):
        best = results[0]

    y_true_binary = _to_binary_target(best.y_true)
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1.2, label='Chance')

    pvalue_col = f'DeLong p vs {best.label}'
    rows: list[dict[str, str]] = []
    for result in results:
        fpr, tpr, _ = roc_curve(y_true_binary, result.y_score_pos)
        if result.label == best.label:
            pvalue_text = '-'
        else:
            pvalue = _delong_pvalue(best.y_true, best.y_score_pos, result.y_score_pos)
            pvalue_text = _format_pvalue(pvalue)

        auc_text = 'nan' if np.isnan(result.full_auc) else f'{result.full_auc:.3f}'
        legend = f'{result.label} (AUC={auc_text}, p={pvalue_text})'
        ax.plot(
            fpr,
            tpr,
            linewidth=2.6 if result.label == best.label else 1.8,
            label=legend,
        )
        rows.append(
            {
                'Model': result.label,
                'AUC': _format_rate(result.full_auc),
                pvalue_col: pvalue_text,
            }
        )

    ax.set(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title=f'{config.title}\nBest model: {best.label}',
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc='lower right', fontsize=9)
    return pd.DataFrame(rows), fig
