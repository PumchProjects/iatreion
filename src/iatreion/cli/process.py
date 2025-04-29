from iatreion.configs import PreprocessorConfig
from iatreion.preprocessors import Preprocessor

from .common import app


@app.command(sort_key=0)
def process(*, config: PreprocessorConfig) -> None:
    """Process the data."""
    preprocessor: Preprocessor = Preprocessor[config.data_name](config)
    preprocessor.process()
