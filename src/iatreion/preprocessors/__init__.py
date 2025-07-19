from typing import cast

from iatreion.configs import DataName, PreprocessorConfig, data_name_mapping

from .base import Preprocessor
from .blood_biomarker import BiomarkerPreprocessor
from .cog_adl import AdlPreprocessor
from .cog_cdr import CdrPreprocessor
from .cog_composite import CompositePreprocessor
from .cog_mmse import MmsePreprocessor
from .cog_mmse_sum import MmseSumPreprocessor
from .cog_moca import MocaPreprocessor
from .cog_moca_sum import MocaSumPreprocessor
from .gene_snp import SnpPreprocessor
from .mri_cbf import CbfPreprocessor
from .mri_csvd import CsvdPreprocessor
from .mri_volume import VolumeAveragePreprocessor, VolumePreprocessor
from .sequential import SequentialPreprocessor


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
        case 'adl':
            return AdlPreprocessor(config)
        case 'adl-sum':
            return AdlPreprocessor(config, is_sum=True)
        case 'screen-sum':
            original_name = config.dataset.name
            children: list[tuple[DataName, Preprocessor]] = []
            for name_ in data_name_mapping[original_name].split(','):
                name = cast(DataName, name_)
                config.dataset.name = name
                children.append((name, get_preprocessor(config)))
            config.dataset.name = original_name
            return SequentialPreprocessor(config, children)
        case 'composite-bin':
            return CompositePreprocessor(config)
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
