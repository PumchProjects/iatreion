import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from iatreion.api import RrlWaterfallBundle

_SUPPORT_COLOR = '#d95f02'
_OPPOSE_COLOR = '#1f77b4'
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


def _draw_module_waterfall(
    ax: plt.Axes,
    contributions: pd.DataFrame,
    module_row: pd.Series,
    *,
    final_label: str,
) -> None:
    rows = contributions.sort_values('Order', ignore_index=True).to_dict('records')
    y_pos = np.arange(len(rows))
    height = 0.65
    all_points = [0.0, float(module_row['Target Margin'])]
    for idx, row in enumerate(rows):
        start = float(row['Start'])
        end = float(row['End'])
        delta = float(row['Signed Score'])
        color = _SUPPORT_COLOR if delta >= 0 else _OPPOSE_COLOR
        left = min(start, end)
        width = abs(delta)
        all_points.extend([start, end])
        ax.barh(
            idx,
            width,
            left=left,
            height=height,
            color=color,
            edgecolor='black',
            linewidth=0.8,
        )
        if idx > 0:
            connector_y = [idx - 1 + height / 2, idx - height / 2]
            ax.plot(
                [start, start],
                connector_y,
                color=_CONNECTOR_COLOR,
                linewidth=1.0,
                linestyle='--',
            )

    x_min = min(all_points)
    x_max = max(all_points)
    span = x_max - x_min
    pad = max(0.25, span * 0.08)
    x_left = x_min - pad
    x_right = x_max + pad
    text_pad = max(0.08, span * 0.02)
    ax.set_xlim(x_left, x_right)

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
        )

    ax.axvline(0.0, color='black', linewidth=1.0, linestyle=':')
    ax.axvline(
        float(module_row['Target Margin']),
        color='#444444',
        linewidth=1.0,
        linestyle='-.',
    )
    ax.set_yticks(y_pos, [_wrap_label(str(row['Display'])) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f'Signed score toward final label "{final_label}"')
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title(
        f'{module_row["Module"]} | pred={module_row["Module Label"]} '
        f'{_format_pct(float(module_row["Module Probability"]))} | '
        f'target={final_label} '
        f'{_format_pct(float(module_row["Target Probability"]))} | '
        f'conf={_format_pct(float(module_row["Confidence"]))} | '
        f'weight={float(module_row["Module Weight"]):.4f}\n'
        f'margin={_format_score(float(module_row["Target Margin"]))} | '
        f'active rules={int(module_row["Active Rule Count"])}',
        fontsize=10,
    )


def rrl_rule_waterfall_plot(
    bundle: RrlWaterfallBundle,
    *,
    title: str = '',
) -> Figure:
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
            f'final={bundle.sample.final_label} '
            f'{_format_pct(bundle.sample.final_probability)}, '
            f'conf={_format_pct(bundle.sample.final_confidence)}'
        ),
        fontsize=12,
    )
    return fig
