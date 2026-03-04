import sys

from cyclopts import App, Parameter
from cyclopts.config import Toml
from rich.console import Console

from iatreion.utils import get_config_path

console = Console()
app = App(
    name='iatreion',
    help='An interpretable dementia diagnoser.',
    default_parameter=Parameter(negative='', parse=r'^(?!_)'),
    config=Toml(get_config_path()),
    console=console,
    help_on_error=True,
)
app['--help'].group = 'Admin'
app['--version'].group = 'Admin'

app.command('iatreion.cli.process:process', sort_key=0)
app.command('iatreion.cli.train:sub_app', name='train')
app.command('iatreion.cli.rrl_eval:rrl_eval', sort_key=2)
app.command('iatreion.cli.show:sub_app', name='show')


def main() -> None:
    try:
        app()
    except Exception:
        console.print_exception()
        sys.exit(1)
