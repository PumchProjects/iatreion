from itertools import count

from cyclopts import App, Group

from iatreion.api import get_rule_waterfall_data
from iatreion.configs import (
    RrlEvalPlotConfig,
    ShowDataConfig,
    ShowImportanceConfig,
    ShowPerformanceConfig,
    ShowShapConfig,
)
from iatreion.show_helpers import (
    acc_mcnemar_ci_barplot,
    acc_wilcoxon_std_barplot,
    auprc_wilcoxon_pvalue_heatmap,
    auprc_wilcoxon_std_barplot,
    auroc_delong_ci_barplot,
    auroc_delong_pvalue_heatmap,
    auroc_wilcoxon_pvalue_heatmap,
    auroc_wilcoxon_std_barplot,
    bar,
    feature_importance_barplot,
    feature_importance_heatmap,
    make_ci_delong_table,
    make_feature_difference_table,
    make_mean_std_wilcoxon_table,
    make_table_1,
    radar,
    roc_delong_comparison_plot,
    rrl_rule_waterfall_plot,
    save,
    shap_dependence_plot,
    shap_summary_plot,
    shap_waterfall_plot,
    violin,
)

from .common import console

sub_app = App(name='show', help='Make figures and tables.')
data = Group.create_ordered('Data')
performance = Group.create_ordered('Performance')
interpretability = Group.create_ordered('Interpretability')
counter = count()


@sub_app.command(group=data, sort_key=next(counter))
def table_1(*, config: ShowDataConfig) -> None:
    """Table 1: Demographics and Clinical Characteristics."""
    table = make_table_1(config)
    console.print(save(config, table))


@sub_app.command(group=data, sort_key=next(counter))
def violin_mmse(*, config: ShowDataConfig) -> None:
    """Violin Plot of MMSE Score."""
    fig = violin(config, 'MMSE', 'MMSE Score')
    save(config, fig=fig)


@sub_app.command(group=data, sort_key=next(counter))
def violin_age(*, config: ShowDataConfig) -> None:
    """Violin Plot of Age."""
    fig = violin(config, 'Age', 'Age (years)')
    save(config, fig=fig)


@sub_app.command(group=data, sort_key=next(counter))
def bar_sex(*, config: ShowDataConfig) -> None:
    """Stacked Bar Chart of Sex."""
    fig = bar(config, 'Sex', ['Female', 'Male'])
    save(config, fig=fig)


@sub_app.command(group=data, sort_key=next(counter))
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


@sub_app.command(group=data, sort_key=next(counter))
def latex_feature_diff(
    *,
    config: ShowDataConfig,
    top_k: int = 20,
    method: str = 'welch',
) -> None:
    """Make a LaTeX table for the most significant
    feature differences in binary groups."""
    table = make_feature_difference_table(config, top_k=top_k, method=method)
    console.print(save(config, table, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def latex_mean_std_wilcoxon(*, config: ShowPerformanceConfig) -> None:
    """Make a LaTeX table for mean/std metrics and AUROC Wilcoxon test."""
    table = make_mean_std_wilcoxon_table(config)
    console.print(save(config, table, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def latex_ci_delong(*, config: ShowPerformanceConfig) -> None:
    """Make a LaTeX table for 95% CI metrics and AUROC DeLong test."""
    table = make_ci_delong_table(config)
    console.print(save(config, table, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def heatmap_auroc_wilcoxon_pvalue(*, config: ShowPerformanceConfig) -> None:
    """Make pairwise AUROC Wilcoxon p-value heatmap for all models."""
    matrix, fig = auroc_wilcoxon_pvalue_heatmap(config)
    console.print(save(config, matrix, fig, float_format=lambda value: f'{value:.4f}'))


@sub_app.command(group=performance, sort_key=next(counter))
def heatmap_auprc_wilcoxon_pvalue(*, config: ShowPerformanceConfig) -> None:
    """Make pairwise AUPRC Wilcoxon p-value heatmap for all models."""
    matrix, fig = auprc_wilcoxon_pvalue_heatmap(config)
    console.print(save(config, matrix, fig, float_format=lambda value: f'{value:.4f}'))


@sub_app.command(group=performance, sort_key=next(counter))
def heatmap_auroc_delong_pvalue(*, config: ShowPerformanceConfig) -> None:
    """Make pairwise AUROC DeLong p-value heatmap for all models."""
    matrix, fig = auroc_delong_pvalue_heatmap(config)
    console.print(save(config, matrix, fig, float_format=lambda value: f'{value:.4f}'))


@sub_app.command(group=performance, sort_key=next(counter))
def roc_delong_comparison(*, config: ShowPerformanceConfig) -> None:
    """Make ROC comparison plot with DeLong p-values in legend."""
    table, fig = roc_delong_comparison_plot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def bar_auroc_delong_ci(*, config: ShowPerformanceConfig) -> None:
    """Bar plot for AUROC + DeLong + 95% CI."""
    table, fig = auroc_delong_ci_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def bar_acc_mcnemar_ci(*, config: ShowPerformanceConfig) -> None:
    """Bar plot for ACC + McNemar + 95% CI."""
    table, fig = acc_mcnemar_ci_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def bar_auroc_wilcoxon_std(*, config: ShowPerformanceConfig) -> None:
    """Bar plot for AUROC + Wilcoxon + fold std."""
    table, fig = auroc_wilcoxon_std_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def bar_auprc_wilcoxon_std(*, config: ShowPerformanceConfig) -> None:
    """Bar plot for AUPRC + Wilcoxon + fold std."""
    table, fig = auprc_wilcoxon_std_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=performance, sort_key=next(counter))
def bar_acc_wilcoxon_std(*, config: ShowPerformanceConfig) -> None:
    """Bar plot for ACC + Wilcoxon + fold std."""
    table, fig = acc_wilcoxon_std_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=interpretability, sort_key=next(counter))
def bar_feature_importance(*, config: ShowImportanceConfig) -> None:
    """Bar plot for aggregated feature importance."""
    table, fig = feature_importance_barplot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=interpretability, sort_key=next(counter))
def heatmap_feature_importance(*, config: ShowImportanceConfig) -> None:
    """Heatmap for aggregated feature importance."""
    table, fig = feature_importance_heatmap(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=interpretability, sort_key=next(counter))
def shap_summary(*, config: ShowShapConfig) -> None:
    """SHAP summary beeswarm plot."""
    table, fig = shap_summary_plot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=interpretability, sort_key=next(counter))
def shap_waterfall(*, config: ShowShapConfig) -> None:
    """SHAP waterfall plot for one sample."""
    table, fig = shap_waterfall_plot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=interpretability, sort_key=next(counter))
def shap_dependence(*, config: ShowShapConfig) -> None:
    """SHAP dependence plot for one feature."""
    table, fig = shap_dependence_plot(config)
    console.print(save(config, table, fig, index=False))


@sub_app.command(group=interpretability, sort_key=next(counter))
def rrl_waterfall(*, config: RrlEvalPlotConfig | None = None) -> None:
    """Plot per-module RRL rule waterfalls for one sample."""
    if config is None:
        config = RrlEvalPlotConfig()

    bundle = get_rule_waterfall_data(config)
    fig = rrl_rule_waterfall_plot(bundle, title=config.title)
    fig.savefig(config.get_output_path('png'), dpi=300)
    bundle.module_table.to_csv(config.get_output_path('tsv'), sep='\t', index=False)
    bundle.contribution_table.to_csv(
        config.get_output_path('rules.tsv'),
        sep='\t',
        index=False,
    )
    console.print(
        'Saved RRL waterfall plot to '
        f'{config.get_output_path("png")} and tables to '
        f'{config.get_output_path("tsv")} / {config.get_output_path("rules.tsv")}'
    )
