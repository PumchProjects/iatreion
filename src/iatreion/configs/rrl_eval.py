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
    names: Annotated[
        list[DataName], Parameter(name=['--names', '-n'], consume_multiple=True)
    ] = field(default_factory=list)
    'Name of the data file.'

    groups: Annotated[
        list[str], Parameter(name=['--groups', '-g'], consume_multiple=True)
    ] = field(default_factory=list)
    'Group names of the data.'

    thesaurus: Annotated[str, Parameter(name=['--thesaurus', '-t'])] = ''
    'Root directory for trained RRL models.'

    process: Annotated[str, Parameter(name=['--process', '-p'])] = ''
    'Path to the processing info file.'

    data: Annotated[dict[str, str], Parameter(name=['--data', '-d'])] = field(
        default_factory=dict
    )
    'Path to the data file.'

    vmri: Annotated[str, Parameter(name=['--vmri', '-v'])] = ''
    'Path to the Vmri_mean_sd data file.'

    vmri_change: Annotated[str, Parameter(name=['--vmri-change', '-vc'])] = ''
    'Path to the Vmri_mean_sd column name change file.'

    mode: Annotated[
        Literal['single', 'batch', 'eval', 'show'], Parameter(name=['--mode', '-m'])
    ] = 'single'
    'Mode of RRL evaluation.'

    keep: Annotated[
        Literal['all', 'first', 'last'], Parameter(name=['--keep', '-k'])
    ] = 'all'
    """Deduplication strategy for duplicated samples when in eval mode.
'first': keep the first sample of each patient.
'last': keep the last sample of each patient.
'all' (default): do not deduplicate samples.
"""

    debug: Annotated[bool, Parameter(name=['--debug', '-D'], negative='')] = False
    'Whether to enable debug mode.'

    def make_configs(self) -> tuple[PreprocessorConfig, DiscreteRrlConfig]:
        # HACK: Empty prefix, set simple to False implicitly
        dataset = DatasetConfig(prefix=Path(), names=self.names)
        train = TrainConfig(
            group_names=self.groups,
            keep=self.keep,
            final=True,
            log_root=Path(self.thesaurus),
        )
        # HACK: Empty input prefix
        process_config = PreprocessorConfig(
            dataset=dataset,
            input_prefix=Path(),
            vmri_data_path_=Path(self.vmri),
            vmri_change_path_=Path(self.vmri_change),
            data_paths={name: Path(path) for name, path in self.data.items()},
            process_info_path_=Path(self.process),
            final=True,
            eval=self.mode == 'eval',
            debug=self.debug,
        )
        rrl_config = DiscreteRrlConfig(dataset=dataset, train=train)
        return process_config, rrl_config
