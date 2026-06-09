import sys
from typing import Annotated

from cyclopts import App, Parameter
from cyclopts.config import Toml
from cyclopts.types import ExistingTomlPath
from rich.console import Console

from iatreion.utils import get_config_path

console = Console()
app = App(
    name='iatreion',
    help='An interpretable dementia diagnoser.',
    default_parameter=Parameter(negative='', parse=r'^(?!_)'),
    console=console,
    help_on_error=True,
)
app['--help'].group = 'Admin'
app['--version'].group = 'Admin'

app.command(
    'iatreion.cli.process:process',
    help='Process the data.',
    sort_key=0,
)
app.command(
    'iatreion.cli.train:sub_app',
    name='train',
    help='Train a model.',
    sort_key=1,
)
app.command(
    'iatreion.cli.evaluate:sub_app',
    name='eval',
    help='Evaluate final models.',
    sort_key=2,
)
app.command(
    'iatreion.cli.show:sub_app',
    name='show',
    help='Make figures and tables.',
    sort_key=3,
)


@app.meta.default
def meta(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    config: Annotated[
        ExistingTomlPath | None, Parameter(help='Path to the TOML config file.')
    ] = None,
) -> None:
    app.config = Toml(config or get_config_path())
    app(tokens)


def main() -> None:
    try:
        app.meta()
    except Exception:
        console.print_exception()
        sys.exit(1)
