import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from iatreion.api import RrlWaterfallBundle
from iatreion.gui.static import names_mapping
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


def _format_rule_label(row: dict[str, object]) -> str:
    display = str(row['Display'])
    module = str(row['Module'])
    if module == 'All':
        return display
    return f'[{names_mapping.get(module, module)}] {display}'


def _draw_global_waterfall(
    ax: plt.Axes,
    contributions: pd.DataFrame,
    *,
    bundle: RrlWaterfallBundle,
) -> None:
    sample = bundle.sample
    target_name = _get_target_name(sample.final_label)
    rows = (
        contributions[contributions['Kind'] != 'Bias']
        .sort_values('Order', ignore_index=True)
        .to_dict('records')
    )
    bias = float(contributions.loc[contributions['Kind'] == 'Bias', 'End'].iloc[0])
    score = sample.final_score
    boundary = sample.final_boundary
    if not rows:
        span = max(abs(score), abs(bias), abs(boundary), 1.0)
        pad = max(0.25, span * 0.08)
        ax.set_xlim(
            min(0.0, bias, score, boundary) - pad,
            max(0.0, bias, score, boundary) + pad,
        )
        ax.set_ylim(1.2, -1.2)
        ax.axvline(boundary, color='black', linewidth=1.1, linestyle='--')
        if not np.isclose(boundary, 0.0):
            ax.axvline(0.0, color='#aaaaaa', linewidth=0.8, linestyle=':')
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
                sample.final_probability,
            ),
            va='bottom',
        )
        ax.text(
            boundary,
            -0.95,
            f'boundary = {_format_score(boundary)}',
            ha='center',
            va='bottom',
            fontsize=9,
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
    all_points = [0.0, bias, score, boundary]
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
            sample.final_probability,
        ),
        va='bottom',
    )
    ax.axvline(boundary, color='black', linewidth=1.1, linestyle='--')
    if not np.isclose(boundary, 0.0):
        ax.axvline(0.0, color='#aaaaaa', linewidth=0.8, linestyle=':')
    ax.text(
        boundary,
        top_text_y + 0.2,
        f'boundary = {_format_score(boundary)}',
        ha='center',
        va='bottom',
        fontsize=9,
    )
    ax.set_yticks(y_pos, [_wrap_label(_format_rule_label(row)) for row in rows])
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
    contribution_table = bundle.contribution_table
    rule_count = len(contribution_table[contribution_table['Kind'] != 'Bias'])
    height = max(4.0, 2.8 + 0.5 * rule_count)
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(13.0, height),
        layout='constrained',
        squeeze=False,
    )
    _draw_global_waterfall(
        axes[0, 0],
        contribution_table,
        bundle=bundle,
    )

    fig.suptitle(
        title
        or (
            'RRL Rule Waterfall\n'
            f'sample={bundle.sample.sample_id}, '
            f'final={target_name} '
            f'{_format_pct(bundle.sample.final_probability)}, '
            f'threshold={_format_pct(bundle.sample.threshold)}'
        ),
        fontsize=12,
    )
    return fig
