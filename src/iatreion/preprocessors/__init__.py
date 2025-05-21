from iatreion.configs import PreprocessorConfig

from .base import Preprocessor
from .gene_snp import SnpPreprocessor


def get_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    match config.dataset.name:
        case 'snp':
            return SnpPreprocessor(config)
        case _:
            raise ValueError(f'Unknown dataset name: {config.dataset.name}')
