from iatreion.configs import PreprocessorConfig

from .base import Preprocessor
from .gene_snp import SnpPreprocessor
from .mri_cbf import CbfPreprocessor
from .mri_volume import VolumePreprocessor


def get_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    match config.dataset.name:
        case 'cbf':
            return CbfPreprocessor(config)
        case 'volume':
            return VolumePreprocessor(config)
        case 'snp':
            return SnpPreprocessor(config)
        case _:
            raise ValueError(f'Unknown dataset name: {config.dataset.name}')
