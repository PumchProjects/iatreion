import os
import sys

from cyclopts import App
from cyclopts.config import Toml
from rich.console import Console

console = Console()
app = App(
    name='iatreion',
    help='An interpretable dementia diagnoser.',
    config=Toml(os.environ.get('IATREION_CONFIG_PATH', 'config.toml')),
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
