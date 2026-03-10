from itertools import count

from cyclopts import App

from iatreion.configs import ShowDataConfig, ShowResultConfig
from iatreion.show_helpers import (
    acc_mcnemar_ci_barplot,
    acc_wilcoxon_std_barplot,
    auc_delong_ci_barplot,
    auc_wilcoxon_std_barplot,
    bar,
    delong_pvalue_heatmap,
    feature_importance_barplot,
    feature_importance_heatmap,
    make_ci_delong_table,
    make_mean_std_wilcoxon_table,
    make_table_1,
    radar,
    roc_delong_comparison_plot,
    save,
    violin,
    wilcoxon_pvalue_heatmap,
)

from .common import console

sub_app = App(name='show', help='Make figures and tables.', sort_key=3)
counter = count()


@sub_app.command(group='Data', sort_key=next(counter))
def table_1(*, config: ShowDataConfig) -> None:
    """Table 1: Demographics and Clinical Characteristics."""
    table = make_table_1(config)
    console.print(save(config, table))


@sub_app.command(group='Data', sort_key=next(counter))
def violin_mmse(*, config: ShowDataConfig) -> None:
    """Violin Plot of MMSE Score."""
    fig = violin(config, 'MMSE', 'MMSE Score')
    save(config, fig=fig)


@sub_app.command(group='Data', sort_key=next(counter))
def violin_age(*, config: ShowDataConfig) -> None:
    """Violin Plot of Age."""
    fig = violin(config, 'Age', 'Age (years)')
    save(config, fig=fig)


@sub_app.command(group='Data', sort_key=next(counter))
def bar_sex(*, config: ShowDataConfig) -> None:
    """Stacked Bar Chart of Sex."""
    fig = bar(config, 'Sex', ['Female', 'Male'])
    save(config, fig=fig)


@sub_app.command(group='Data', sort_key=next(counter))
def radar_mmse(*, config: ShowDataConfig) -> None:
    """Radar Chart of MMSE Subdomains."""
    domains = [
        'Orientation',
        'Registration',
        'Attention & Calculation',
        'Recall',
        'Language',
    ]
    fig = radar(config, domains)
    save(config, fig=fig)


@sub_app.command(group='Result', sort_key=next(counter))
def latex_mean_std_wilcoxon(*, config: ShowResultConfig) -> None:
    """Make a LaTeX table for mean/std metrics and Wilcoxon test."""
    table = make_mean_std_wilcoxon_table(config)
    console.print(save(config, table, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def latex_ci_delong(*, config: ShowResultConfig) -> None:
    """Make a LaTeX table for 95% CI metrics and DeLong test."""
    table = make_ci_delong_table(config)
    console.print(save(config, table, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def heatmap_wilcoxon_pvalue(*, config: ShowResultConfig) -> None:
    """Make pairwise Wilcoxon p-value heatmap for all models."""
    matrix, fig = wilcoxon_pvalue_heatmap(config)
    console.print(save(config, matrix, fig, float_format=lambda value: f'{value:.4f}'))


@sub_app.command(group='Result', sort_key=next(counter))
def heatmap_delong_pvalue(*, config: ShowResultConfig) -> None:
    """Make pairwise DeLong p-value heatmap for all models."""
    matrix, fig = delong_pvalue_heatmap(config)
    console.print(save(config, matrix, fig, float_format=lambda value: f'{value:.4f}'))


@sub_app.command(group='Result', sort_key=next(counter))
def roc_delong_comparison(*, config: ShowResultConfig) -> None:
    """Make ROC comparison plot with DeLong p-values in legend."""
    table, fig = roc_delong_comparison_plot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def bar_auc_delong_ci(*, config: ShowResultConfig) -> None:
    """Bar plot for AUC + DeLong + 95% CI."""
    table, fig = auc_delong_ci_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def bar_acc_mcnemar_ci(*, config: ShowResultConfig) -> None:
    """Bar plot for ACC + McNemar + 95% CI."""
    table, fig = acc_mcnemar_ci_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def bar_auc_wilcoxon_std(*, config: ShowResultConfig) -> None:
    """Bar plot for AUC + Wilcoxon + fold std."""
    table, fig = auc_wilcoxon_std_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def bar_acc_wilcoxon_std(*, config: ShowResultConfig) -> None:
    """Bar plot for ACC + Wilcoxon + fold std."""
    table, fig = acc_wilcoxon_std_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def bar_feature_importance(*, config: ShowResultConfig) -> None:
    """Bar plot for aggregated feature importance."""
    table, fig = feature_importance_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group='Result', sort_key=next(counter))
def heatmap_feature_importance(*, config: ShowResultConfig) -> None:
    """Heatmap for aggregated feature importance."""
    table, fig = feature_importance_heatmap(config)
    console.print(save(config, table, fig, index=False))
