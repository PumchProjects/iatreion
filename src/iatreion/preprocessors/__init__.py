from iatreion.configs import PreprocessorConfig

from .base import Preprocessor
from .blood_biomarker import BiomarkerPreprocessor
from .cog_cdr import CdrPreprocessor
from .cog_mmse import MmsePreprocessor
from .cog_mmse_sum import MmseSumPreprocessor
from .cog_moca import MocaPreprocessor
from .cog_moca_sum import MocaSumPreprocessor
from .gene_snp import SnpPreprocessor
from .mri_cbf import CbfPreprocessor
from .mri_csvd import CsvdPreprocessor
from .mri_volume import VolumeAveragePreprocessor, VolumePreprocessor


def get_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    match config.dataset.name:
        case 'cdr':
            return CdrPreprocessor(config)
        case 'mmse':
            return MmsePreprocessor(config)
        case 'mmse-sum':
            return MmseSumPreprocessor(config)
        case 'moca':
            return MocaPreprocessor(config)
        case 'moca-sum':
            return MocaSumPreprocessor(config)
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
