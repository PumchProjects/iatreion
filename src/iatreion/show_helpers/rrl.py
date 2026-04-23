import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from iatreion.api import RrlWaterfallBundle
from iatreion.show_helpers.data import group_mapping

_SUPPORT_COLOR = (1.0, 0.0, 0.31796406298163893)
_OPPOSE_COLOR = (0.0, 0.5433775692459109, 0.983379062301401)
_CONNECTOR_COLOR = '#888888'


def _wrap_label(text: str, *, width: int = 48) -> str:
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _format_pct(value: float) -> str:
    return 'nan' if np.isnan(value) else f'{value:.1%}'


def _format_score(value: float) -> str:
    return 'nan' if np.isnan(value) else f'{value:+.2f}'


def _format_score_probability(score: float, probability: float) -> str:
    return f'score = {_format_score(score)}, p = {_format_pct(probability)}'


def _get_target_name(label: str) -> str:
    return group_mapping.get(str(label), str(label))


def _get_bar_color(delta: float) -> tuple[float, float, float]:
    return _SUPPORT_COLOR if delta >= 0 else _OPPOSE_COLOR


def _get_head_length(
    *,
    span: float,
    delta: float,
) -> float:
    return min(abs(delta), max(0.025, span * 0.035))


def _draw_arrow_bar(
    ax: plt.Axes,
    *,
    y: float,
    start: float,
    end: float,
    height: float,
    span: float,
    color: tuple[float, float, float],
) -> None:
    delta = end - start
    if np.isclose(delta, 0.0):
        return
    ax.arrow(
        start,
        y,
        delta,
        0.0,
        width=height,
        head_width=height,
        head_length=_get_head_length(span=span, delta=delta),
        length_includes_head=True,
        overhang=0.0,
        fc=color,
        ec=color,
        linewidth=0.0,
        zorder=3,
    )


def _draw_endpoint_annotation(
    ax: plt.Axes,
    *,
    x: float,
    bar_edge_y: float,
    text_y: float,
    text: str,
    va: str,
) -> None:
    guide_gap = 0.12 if va == 'bottom' else -0.12
    ax.plot(
        [x, x],
        [bar_edge_y, text_y + guide_gap],
        color=_CONNECTOR_COLOR,
        linewidth=1.0,
        linestyle='--',
    )
    ax.text(
        x,
        text_y,
        text,
        ha='center',
        va=va,
        fontsize=9,
        fontweight='bold',
    )


def _draw_module_waterfall(
    ax: plt.Axes,
    contributions: pd.DataFrame,
    module_row: pd.Series,
    *,
    final_label: str,
) -> None:
    target_name = _get_target_name(final_label)
    rows = (
        contributions[contributions['Kind'] != 'Bias']
        .sort_values('Order', ignore_index=True)
        .to_dict('records')
    )
    if not rows:
        score = float(module_row['Target Margin'])
        bias = float(module_row['Bias Signed Score'])
        span = max(abs(score), abs(bias), 1.0)
        pad = max(0.25, span * 0.08)
        ax.set_xlim(min(bias, score) - pad, max(bias, score) + pad)
        ax.set_ylim(1.2, -1.2)
        ax.axvline(0.0, color='black', linewidth=0.8, linestyle=':')
        ax.set_yticks([])
        ax.grid(axis='x', linestyle=':', alpha=0.35)
        ax.set_axisbelow(True)
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(
            f'Signed score toward final label "{target_name}"',
            labelpad=10,
        )
        _draw_endpoint_annotation(
            ax,
            x=bias,
            bar_edge_y=0.3,
            text_y=0.95,
            text=f'bias = {_format_score(bias)}',
            va='top',
        )
        _draw_endpoint_annotation(
            ax,
            x=score,
            bar_edge_y=-0.3,
            text_y=-0.95,
            text=_format_score_probability(
                score,
                float(module_row['Target Probability']),
            ),
            va='bottom',
        )
        ax.text(
            0.5,
            0.5,
            'No active rules',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=10,
            style='italic',
        )
        ax.tick_params(axis='y', which='both', length=0)
        return

    y_pos = np.arange(len(rows))
    height = 0.65
    bias = float(module_row['Bias Signed Score'])
    score = float(module_row['Target Margin'])
    all_points = [0.0, bias, score]
    for row in rows:
        all_points.extend([float(row['Start']), float(row['End'])])
    x_min = min(all_points)
    x_max = max(all_points)
    span = x_max - x_min
    pad = max(0.25, span * 0.08)
    x_left = x_min - pad
    x_right = x_max + pad
    text_pad = max(0.08, span * 0.02)
    ax.set_xlim(x_left, x_right)

    for idx, row in enumerate(rows[:-1]):
        connector_x = float(row['Start'])
        connector_y = [idx + height / 2, idx + 1 + height / 2]
        ax.plot(
            [connector_x, connector_x],
            connector_y,
            color=_CONNECTOR_COLOR,
            linewidth=1.0,
            linestyle='--',
            zorder=1,
        )

    for idx, row in enumerate(rows):
        start = float(row['Start'])
        end = float(row['End'])
        delta = float(row['Signed Score'])
        _draw_arrow_bar(
            ax,
            y=idx,
            start=start,
            end=end,
            height=height,
            span=max(span, 1.0),
            color=_get_bar_color(delta),
        )

    for idx, row in enumerate(rows):
        end = float(row['End'])
        delta = float(row['Signed Score'])
        label_x = end + text_pad if delta >= 0 else end - text_pad
        ax.text(
            label_x,
            idx,
            _format_score(delta),
            ha='left' if delta >= 0 else 'right',
            va='center',
            fontsize=9,
            fontweight='bold',
            color=_get_bar_color(delta),
        )

    top_text_y = -1.0
    bottom_text_y = len(rows) - 1 + 1.0
    _draw_endpoint_annotation(
        ax,
        x=bias,
        bar_edge_y=len(rows) - 1 + height / 2,
        text_y=bottom_text_y,
        text=f'bias = {_format_score(bias)}',
        va='top',
    )
    _draw_endpoint_annotation(
        ax,
        x=score,
        bar_edge_y=height / 2,
        text_y=top_text_y,
        text=_format_score_probability(
            score,
            float(module_row['Target Probability']),
        ),
        va='bottom',
    )
    ax.axvline(0.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_yticks(y_pos, [_wrap_label(str(row['Display'])) for row in rows])
    ax.set_ylim(bottom_text_y + 0.25, top_text_y - 0.25)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(
        f'Signed score toward final label "{target_name}"',
        labelpad=10,
    )
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def rrl_rule_waterfall_plot(
    bundle: RrlWaterfallBundle,
    *,
    title: str = '',
) -> Figure:
    target_name = _get_target_name(bundle.sample.final_label)
    module_table = bundle.module_table.reset_index(drop=True)
    contribution_table = bundle.contribution_table
    heights = [
        max(
            3.2,
            1.8
            + 0.5 * len(contribution_table[contribution_table['Module'] == row.Module]),
        )
        for row in module_table.itertuples()
    ]
    fig, axes = plt.subplots(
        len(module_table),
        1,
        figsize=(12.0, sum(heights) + 1.2),
        layout='constrained',
        squeeze=False,
    )
    for ax, (_, module_row) in zip(
        axes[:, 0],
        module_table.iterrows(),
        strict=True,
    ):
        module_contrib = contribution_table[
            contribution_table['Module'] == module_row['Module']
        ]
        _draw_module_waterfall(
            ax,
            module_contrib,
            module_row,
            final_label=bundle.sample.final_label,
        )

    fig.suptitle(
        title
        or (
            'RRL Rule Waterfall\n'
            f'sample={bundle.sample.sample_id}, '
            f'final={target_name} '
            f'{_format_pct(bundle.sample.final_probability)}, '
            f'conf={_format_pct(bundle.sample.final_confidence)}'
        ),
        fontsize=12,
    )
    return fig
