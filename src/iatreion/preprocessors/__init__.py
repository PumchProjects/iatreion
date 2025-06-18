from iatreion.configs import PreprocessorConfig

from .base import Preprocessor
from .blood_biomarker import BiomarkerPreprocessor
from .exam_cdr import CdrPreprocessor
from .gene_snp import SnpPreprocessor
from .mri_cbf import CbfPreprocessor
from .mri_csvd import CsvdPreprocessor
from .mri_volume import VolumeAveragePreprocessor, VolumePreprocessor


def get_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    match config.dataset.name:
        case 'cdr':
            return CdrPreprocessor(config)
        case 'biomarker':
            return BiomarkerPreprocessor(config)
        case 'cbf':
            return CbfPreprocessor(config)
        case 'csvd':
            return CsvdPreprocessor(config)
        case 'volume':
            return VolumePreprocessor(config)
        case 'volume-v':
            return VolumeAveragePreprocessor(config, feature='v')
        case 'volume-pct':
            return VolumeAveragePreprocessor(config, feature='pct')
        case 'snp':
            return SnpPreprocessor(config)
        case _:
            raise ValueError(f'Unknown dataset name: {config.dataset.name}')
