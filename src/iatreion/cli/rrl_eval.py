from pathlib import Path

from iatreion.configs import DiscreteRrlConfig, PreprocessorConfig
from iatreion.models import DiscreteRrlModel
from iatreion.preprocessors import get_preprocessor

from .common import app


@app.command(sort_key=2)
def rrl_eval(*, config: DiscreteRrlConfig) -> None:
    """Evaluate an RRL model."""
    config.dataset.simple = True
    config.train.final = True
    process_config = PreprocessorConfig(dataset=config.dataset, output_prefix=Path())
    preprocessor = get_preprocessor(process_config)
    data = preprocessor.get_data()
    model = DiscreteRrlModel(config)
    print(model.eval(data))
