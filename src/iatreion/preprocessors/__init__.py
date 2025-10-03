from iatreion.configs import PreprocessorConfig

from .base import NamedPreprocessor, Preprocessor
from .blood_biomarker import BiomarkerPreprocessor
from .cog_adl import AdlPreprocessor
from .cog_associative import AssociativeLearningPreprocessor
from .cog_avlt import AvltPreprocessor
from .cog_cdr import CdrPreprocessor
from .cog_composite import CompositePreprocessor
from .cog_episodic import EpisodicMemoryPreprocessor
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
from .reference import ReferencePreprocessor
from .sequential import SequentialPreprocessor


def get_single_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    match config.dataset.name:
        case (
            'life'
            | 'diet-medication'
            | 'family-history'
            | 'medical-history'
            | 'symptom'
        ):
            return HistoryPreprocessor(config, config.dataset.name)
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
        case 'associative-learning':
            return AssociativeLearningPreprocessor(config)
        case 'episodic-memory':
            return EpisodicMemoryPreprocessor(config)
        case 'avlt':
            return AvltPreprocessor(config)
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
        case 'volume-v-nz':
            return VolumeAverageNewPreprocessor(config, feature='v')
        case 'volume-pct-nz':
            return VolumeAverageNewPreprocessor(config, feature='pct')
        case 'snp':
            return SnpPreprocessor(config)
        case name:
            children: list[NamedPreprocessor] = []
            for child_name in config.children_names:
                # HACK: Essential for recursive sequential data
                # HACK: Ensures that config.children_names works correctly
                config.dataset.name = child_name
                children.append((child_name, get_single_preprocessor(config)))
            if len(children) == 0:
                raise ValueError(f'Unknown dataset name: {name}')
            config.dataset.name = name
            return SequentialPreprocessor(config, children)


def get_preprocessor(config: PreprocessorConfig) -> Preprocessor:
    if config.dataset.ref_name is None:
        return get_single_preprocessor(config)
    else:
        name, ref_name = config.dataset.name, config.dataset.ref_name
        child = (name, get_single_preprocessor(config))
        config.dataset.name = ref_name
        ref_child = (ref_name, get_single_preprocessor(config))
        # HACK: Ensures that data are stored in the correct directory
        config.dataset.name = name
        return ReferencePreprocessor(config, child, ref_child)
