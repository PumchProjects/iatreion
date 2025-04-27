import sys

from cyclopts import App
from rich.console import Console

console = Console()
app = App(
    help='An interpretable dementia diagnoser.', console=console, help_on_error=True
)
app['--help'].group = 'Admin'
app['--version'].group = 'Admin'


def main() -> None:
    try:
        app()
    except Exception:
        console.print_exception()
        sys.exit(1)
