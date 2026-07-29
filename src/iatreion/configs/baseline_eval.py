from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from cyclopts import Parameter
from cyclopts.types import PositiveInt
from cyclopts.validators import Number

from .dataset import DataName, DatasetConfig
from .model_base import ModelConfig
from .preprocessor import PreprocessorConfig
from .train import UNUSED_LABEL_NAME, TrainConfig

type BaselineEvalMode = Literal['batch', 'eval']


@Parameter(name='*')
@dataclass(kw_only=True)
class BaselineEvalConfig:
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

    log_root: Path = Path('logs')
    'Root directory for trained final baseline models.'

    process: Annotated[str, Parameter(alias='-p')] = ''
    'Path to the processing info file.'

    data: dict[str, str] = field(default_factory=dict)
    'Path to the external data file.'

    data_sheets: dict[str, str] = field(default_factory=dict)
    'Excel sheet names or indices keyed by raw data name. If not set, use sheet 0.'

    vmri: Annotated[str, Parameter(alias='-v')] = ''
    'Path to the Vmri_mean_sd data file.'

    vmri_change: str = ''
    'Path to the Vmri_mean_sd column name change file.'

    mode: Annotated[BaselineEvalMode, Parameter(alias='-m')] = 'eval'
    'Mode of baseline evaluation.'

    output: Annotated[str, Parameter(alias='-o')] = ''
    'Output path for exported batch predictions.'

    keep: Annotated[Literal['first', 'last'], Parameter(alias='-k')] = 'last'
    'Deduplication strategy for duplicated samples.'

    suspected_case: bool = False
    'Whether to include suspected cases in evaluation.'

    bootstrap_samples: PositiveInt = 1000
    'Number of bootstrap resamples used to estimate confidence intervals.'

    ci_level: Annotated[float, Parameter(validator=Number(gt=0, lt=1))] = 0.95
    'Confidence level in (0, 1) for bootstrap confidence intervals.'

    index_name: str = ''
    'Index column name in the data files.'

    label_name: str = ''
    'Label column name in the data files. Required for eval mode.'

    debug: Annotated[bool, Parameter(alias='-D')] = False
    'Whether to enable debug mode.'

    def _make_configs(
        self,
        model_config_cls: type[ModelConfig],
        **model_config_kwargs: Any,
    ) -> tuple[PreprocessorConfig, ModelConfig]:
        if not self.index_name:
            raise ValueError('index_name is required.')
        if self.mode == 'eval' and not self.label_name:
            raise ValueError('label_name is required for eval mode.')
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
            aggregate='calibrated-fusion',
            bootstrap_samples=self.bootstrap_samples,
            ci_level=self.ci_level,
            log_root=self.log_root,
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
        return process_config, model_config_cls(
            dataset=dataset,
            train=train,
            **model_config_kwargs,
        )

    def make_configs(
        self, model_config_cls: type[ModelConfig]
    ) -> tuple[PreprocessorConfig, ModelConfig]:
        return self._make_configs(model_config_cls)
