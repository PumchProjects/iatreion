from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

from .dataset import DataName, DatasetConfig
from .model_rrl_discrete import DiscreteRrlConfig
from .preprocessor import PreprocessorConfig
from .train import TrainConfig

type ZeroMeanFallback = Literal['uniform', 'bias']


@Parameter(name='*')
@dataclass(kw_only=True)
class RrlEvalConfig:
    names: Annotated[list[DataName], Parameter(alias='-n', consume_multiple=True)] = (
        field(default_factory=list)
    )
    'Names of the data files.'

    groups: Annotated[list[str], Parameter(alias='-g', consume_multiple=True)] = field(
        default_factory=list
    )
    'Group names of the data.'

    thesaurus: Annotated[str, Parameter(alias='-t')] = ''
    'Root directory for trained RRL models.'

    process: Annotated[str, Parameter(alias='-p')] = ''
    'Path to the processing info file.'

    data: dict[str, str] = field(default_factory=dict)
    'Path to the data file.'

    data_sheets: dict[str, str] = field(default_factory=dict)
    'Excel sheet names or indices keyed by raw data name. If not set, use sheet 0.'

    vmri: Annotated[str, Parameter(alias='-v')] = ''
    'Path to the Vmri_mean_sd data file.'

    vmri_change: str = ''
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

    suspected_case: bool = False
    'Whether to include suspected cases in evaluation.'

    index_name: str = ''
    'Index column name in the data files. If not set, use default index name.'

    label_name: str = ''
    'Label column name in the data files. If not set, determined automatically.'

    enabled_biases: dict[str, bool] = field(default_factory=dict)
    'Per-module switches for RRL bias terms. Unspecified modules keep the bias enabled.'

    enabled_rules: dict[str, list[int]] = field(default_factory=dict)
    'Per-module enabled RRL rule indices. Unspecified modules use all rules.'

    zero_mean_fallback: ZeroMeanFallback = 'uniform'
    'How to resolve samples whose enabled RRL terms produce zero scores.'

    sample_id: str = ''
    'Sample ID taken from the dataframe index.'

    top_k: int = 20
    'Number of active rules to display in the RRL waterfall plot.'

    debug: Annotated[bool, Parameter(alias='-D')] = False
    'Whether to enable debug mode.'

    def make_configs(self) -> tuple[PreprocessorConfig, DiscreteRrlConfig]:
        # HACK: Empty prefix
        dataset = DatasetConfig(prefix=Path(), names=self.names)
        train = TrainConfig(
            group_names=self.groups,
            keep=self.keep,
            final=True,
            suspected_case=self.suspected_case,
            label_name=self.label_name or None,
            log_root=Path(self.thesaurus),
            _shuffle=False,
        )
        process_config = PreprocessorConfig(
            dataset=dataset,
            data={name: Path(path) for name, path in self.data.items()},
            data_sheets=self.data_sheets,
            index_name_=self.index_name or None,
            group_columns_=[self.label_name] if self.label_name else None,
            vmri=Path(self.vmri) if self.vmri else None,
            vmri_change=Path(self.vmri_change) if self.vmri_change else None,
            _process_info_path=Path(self.process) if self.process else None,
            _final=True,
            _keep=self.keep,
        )
        rrl_config = DiscreteRrlConfig(dataset=dataset, train=train)
        return process_config, rrl_config
