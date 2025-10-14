from iatreion.configs import PreprocessorConfig
from iatreion.preprocessors import get_preprocessors

from .common import app


@app.command(sort_key=0)
def process(*, config: PreprocessorConfig) -> None:
    """Process the data."""
    preprocessors = get_preprocessors(config)
    for preprocessor in preprocessors:
        preprocessor.process()
    config.save_process_info_dict()
