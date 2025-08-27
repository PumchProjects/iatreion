import sys

from cyclopts import App
from cyclopts.config import Toml
from rich.console import Console

from iatreion.utils import get_config_path

console = Console()
app = App(
    name='iatreion',
    help='An interpretable dementia diagnoser.',
    config=Toml(get_config_path()),
    console=console,
    help_on_error=True,
)
app['--help'].group = 'Admin'
app['--version'].group = 'Admin'


def main() -> None:
    try:
        app()
    except Exception:
        console.print_exception()
        sys.exit(1)
