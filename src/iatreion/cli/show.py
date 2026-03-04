from itertools import count

from cyclopts import App

from iatreion.api import bar, make_table_1, radar, violin
from iatreion.configs import ShowConfig

from .common import console

sub_app = App(name='show', help='Make figures and tables.', sort_key=3)
counter = count()


@sub_app.command(sort_key=next(counter))
def table_1(*, config: ShowConfig) -> None:
    """Table 1: Demographics and Clinical Characteristics."""
    table = make_table_1(config)
    console.print(table)
    table.to_latex(config.root / 'table_1.tex', escape=True)


@sub_app.command(sort_key=next(counter))
def violin_mmse(*, config: ShowConfig) -> None:
    """Violin Plot of MMSE Score."""
    fig = violin(config, 'MMSE', 'MMSE Score')
    fig.savefig(config.root / 'violin_mmse.png', dpi=300)


@sub_app.command(sort_key=next(counter))
def violin_age(*, config: ShowConfig) -> None:
    """Violin Plot of Age."""
    fig = violin(config, 'Age', 'Age (years)')
    fig.savefig(config.root / 'violin_age.png', dpi=300)


@sub_app.command(sort_key=next(counter))
def bar_sex(*, config: ShowConfig) -> None:
    """Stacked Bar Chart of Sex."""
    fig = bar(config, 'Sex', ['Female', 'Male'])
    fig.savefig(config.root / 'bar_sex.png', dpi=300)


@sub_app.command(sort_key=next(counter))
def radar_mmse(*, config: ShowConfig) -> None:
    """Radar Chart of MMSE Subdomains."""
    domains = [
        'Orientation',
        'Registration',
        'Attention & Calculation',
        'Recall',
        'Language',
    ]
    fig = radar(config, domains)
    fig.savefig(config.root / 'radar_mmse.png', dpi=300)
