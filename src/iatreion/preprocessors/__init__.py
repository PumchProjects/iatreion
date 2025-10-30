from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor
from .basic import BasicPreprocessor
from .blood_biomarker import BiomarkerPreprocessor
from .cog_adl import AdlPreprocessor
from .cog_associative import AssociativeLearningPreprocessor
from .cog_avlt import AvltPreprocessor
from .cog_cdr import CdrPreprocessor
from .cog_composite import CompositePreprocessor
from .cog_episodic import EpisodicMemoryPreprocessor
from .cog_had import HadPreprocessor
from .cog_mmse import MmsePreprocessor
from .cog_mmse_sum import MmseSumPreprocessor
from .cog_moca import MocaPreprocessor
from .cog_moca_sum import MocaSumPreprocessor
from .gene_snp import SnpPreprocessor
from .history import HistoryPreprocessor
from .mri_cbf import CbfPreprocessor
from .mri_csvd import CsvdPreprocessor
from .mri_volume import (
    VolumeAverageNewPreprocessor,
    VolumeAveragePreprocessor,
    VolumePreprocessor,
)
from .sequential import SequentialPreprocessor


def get_single_preprocessor(config: PreprocessorConfig, name: DataName) -> Preprocessor:
    stem = config.get_stem(name)
    match stem:
        case 'basic-noage':
            return BasicPreprocessor(config, name, age=False)
        case 'basic':
            return BasicPreprocessor(config, name)
        case (
            'life'
            | 'diet-medication'
            | 'family-history'
            | 'medical-history'
            | 'symptom'
        ):
            return HistoryPreprocessor(config, name)
        case 'cdr':
            return CdrPreprocessor(config, name)
        case 'mmse':
            return MmsePreprocessor(config, name)
        case 'mmse-sum':
            return MmseSumPreprocessor(config, name)
        case 'moca':
            return MocaPreprocessor(config, name)
        case 'moca-sum':
            return MocaSumPreprocessor(config, name)
        case 'adl' | 'adl-sum':
            return AdlPreprocessor(config, name)
        case 'had' | 'had-sum':
            return HadPreprocessor(config, name)
        case 'associative-learning':
            return AssociativeLearningPreprocessor(config, name)
        case 'episodic-memory':
            return EpisodicMemoryPreprocessor(config, name)
        case 'avlt':
            return AvltPreprocessor(config, name)
        case 'composite-bin':
            return CompositePreprocessor(config, name)
        case 'biomarker':
            return BiomarkerPreprocessor(config, name)
        case 'cbf':
            return CbfPreprocessor(config, name)
        case 'csvd':
            return CsvdPreprocessor(config, name)
        case 'volume':
            return VolumePreprocessor(config, name)
        case 'volume-v' | 'volume-pct':
            return VolumeAveragePreprocessor(config, name)
        case 'volume-z-v' | 'volume-z-pct':
            return VolumeAveragePreprocessor(config, name, use_z=True)
        case 'volume-nz-v' | 'volume-nz-pct':
            return VolumeAverageNewPreprocessor(config, name)
        case 'volume-new-v' | 'volume-new-pct' | 'volume-adni-v' | 'volume-adni-pct':
            return VolumeAverageNewPreprocessor(config, name, new=True)
        case 'snp':
            return SnpPreprocessor(config, name)
        case _:
            children: list[Preprocessor] = []
            for child_name in config.children_names(name):
                children.append(get_single_preprocessor(config, child_name))
            if len(children) == 0:
                raise ValueError(f'Unknown dataset name: {name}')
            return SequentialPreprocessor(config, name, children)


def get_preprocessors(config: PreprocessorConfig) -> list[Preprocessor]:
    return [get_single_preprocessor(config, name) for name in config.dataset.names]
