from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

from .dataset import DataName, DatasetConfig
from .model_rrl_discrete import DiscreteRrlConfig
from .preprocessor import PreprocessorConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class RrlEvalConfig:
    names: Annotated[list[DataName], Parameter(alias='-n', consume_multiple=True)] = (
        field(default_factory=list)
    )
    'Name of the data file.'

    groups: Annotated[list[str], Parameter(alias='-g', consume_multiple=True)] = field(
        default_factory=list
    )
    'Group names of the data.'

    thesaurus: Annotated[str, Parameter(alias='-t')] = ''
    'Root directory for trained RRL models.'

    process: Annotated[str, Parameter(alias='-p')] = ''
    'Path to the processing info file.'

    data: Annotated[dict[str, str], Parameter(alias='-d')] = field(default_factory=dict)
    'Path to the data file.'

    vmri: Annotated[str, Parameter(alias='-v')] = ''
    'Path to the Vmri_mean_sd data file.'

    vmri_change: Annotated[str, Parameter(alias='-vc')] = ''
    'Path to the Vmri_mean_sd column name change file.'

    mode: Annotated[
        Literal['single', 'batch', 'eval', 'show'], Parameter(alias='-m')
    ] = 'single'
    'Mode of RRL evaluation.'

    keep: Annotated[Literal['first', 'last'], Parameter(alias='-k')] = 'last'
    """Deduplication strategy for duplicated samples.
'first': keep the first sample of each patient.
'last': keep the last sample of each patient.
"""

    suspected_case: Annotated[bool, Parameter(alias='-sc', negative='')] = False
    'Whether to include suspected cases in evaluation.'

    index_name: Annotated[str, Parameter(alias='-in')] = ''
    'Index column name in the data files. If not set, use default index name.'

    label_name: Annotated[str, Parameter(alias='-ln')] = ''
    'Label column name in the data files. If not set, determined automatically.'

    debug: Annotated[bool, Parameter(alias='-D', negative='')] = False
    'Whether to enable debug mode.'

    def make_configs(self) -> tuple[PreprocessorConfig, DiscreteRrlConfig]:
        # HACK: Empty prefix, set simple to False implicitly
        dataset = DatasetConfig(prefix=Path(), names=self.names)
        train = TrainConfig(
            group_names=self.groups,
            keep=self.keep,
            final=True,
            suspected_case=self.suspected_case,
            label_name=self.label_name or None,
            log_root=Path(self.thesaurus),
        )
        # HACK: Empty input prefix
        process_config = PreprocessorConfig(
            dataset=dataset,
            input_prefix=Path(),
            index_name_=self.index_name or None,
            group_columns_=[self.label_name] if self.label_name else None,
            vmri_data_path_=Path(self.vmri),
            vmri_change_path_=Path(self.vmri_change),
            data_paths={name: Path(path) for name, path in self.data.items()},
            process_info_path_=Path(self.process),
            final=True,
        )
        rrl_config = DiscreteRrlConfig(dataset=dataset, train=train)
        return process_config, rrl_config
