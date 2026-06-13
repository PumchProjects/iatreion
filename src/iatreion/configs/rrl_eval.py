from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import PositiveInt
from cyclopts.validators import Number

from .dataset import DataName, DatasetConfig
from .model_rrl_discrete import DiscreteRrlConfig
from .preprocessor import PreprocessorConfig
from .train import UNUSED_LABEL_NAME, TrainConfig

type ZeroMeanFallback = Literal['uniform', 'bias']
type RrlEvalMode = Literal['single', 'batch', 'eval', 'show', 'rule-or']


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

    positive_label: str = ''
    'Positive class label for binary tasks.'

    log_root: str = ''
    'Root directory for trained final RRL models.'

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

    mode: Annotated[RrlEvalMode, Parameter(alias='-m')] = 'single'
    'Mode of RRL evaluation.'

    output: Annotated[str, Parameter(alias='-o')] = ''
    'Output path for exported batch results or rule-OR tables.'

    keep: Annotated[Literal['first', 'last'], Parameter(alias='-k')] = 'last'
    """Deduplication strategy for duplicated samples.
'first': keep the first sample of each patient.
'last': keep the last sample of each patient.
"""

    suspected_case: bool = False
    'Whether to include suspected cases in evaluation.'

    bootstrap_samples: PositiveInt = 1000
    'Number of bootstrap resamples used to estimate confidence intervals.'

    ci_level: Annotated[float, Parameter(validator=Number(gt=0, lt=1))] = 0.95
    'Confidence level in (0, 1) for bootstrap confidence intervals.'

    index_name: str = ''
    'Index column name in the data files. Required for modes that read external data.'

    label_name: str = ''
    'Label column name in the data files. Required for eval and rule-or modes.'

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
        if self.mode != 'show' and not self.index_name:
            raise ValueError(
                'index_name is required for modes that read external data.'
            )
        if self.mode in {'eval', 'rule-or'} and not self.label_name:
            raise ValueError('label_name is required for eval and rule-or modes.')
        label_name = self.label_name or UNUSED_LABEL_NAME
        group_columns = [self.label_name] if self.label_name else []
        dataset = DatasetConfig(prefix=Path(), names=self.names)
        train = TrainConfig(
            group_names=self.groups,
            keep=self.keep,
            final=True,
            suspected_case=self.suspected_case,
            label_name=label_name,
            positive_label=self.positive_label,
            bootstrap_samples=self.bootstrap_samples,
            ci_level=self.ci_level,
            log_root=Path(self.log_root),
            _shuffle=False,
        )
        process_config = PreprocessorConfig(
            dataset=dataset,
            data={name: Path(path) for name, path in self.data.items()},
            data_sheets=self.data_sheets,
            index_name=self.index_name,
            group_columns=group_columns,
            vmri=Path(self.vmri) if self.vmri else None,
            vmri_change=Path(self.vmri_change) if self.vmri_change else None,
            _process_info_path=Path(self.process) if self.process else None,
            _final=True,
            _keep=self.keep,
        )
        rrl_config = DiscreteRrlConfig(dataset=dataset, train=train)
        return process_config, rrl_config
