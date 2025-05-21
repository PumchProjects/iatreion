from iatreion.configs import PreprocessorConfig
from iatreion.preprocessors import get_preprocessor

from .common import app


@app.command(sort_key=0)
def process(*, config: PreprocessorConfig) -> None:
    """Process the data."""
    preprocessor = get_preprocessor(config)
    preprocessor.process()
