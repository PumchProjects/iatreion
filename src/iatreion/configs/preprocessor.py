from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter

from .common import CommonConfig, DataName

data_name_mapping: dict[DataName, str] = {
    'snp': '基因_snp.csv',
}


@Parameter(name='*')
@dataclass
class PreprocessorConfig:
    # HACK: Cyclopts does not support dataclass inheritance yet
    # HACK: The help texts of the parameters are not shown in the help message
    # HACK: Below is the current workaround
    # HACK: Alternative workaround: use Parameter(help=...)
    # TODO: Read Cyclopts documentation to find a better way
    common: CommonConfig

    # TODO: Add more parameters for the preprocessor, e.g. filling missing values

    def __post_init__(self) -> None:
        self.common.output_prefix.mkdir(parents=True, exist_ok=True)

    @property
    def group_data_path(self) -> Path:
        return self.common.data_prefix / '患者及分组加密对应表.xlsx'

    @property
    def data_path(self) -> Path:
        return self.common.data_prefix / data_name_mapping[self.common.data_name]

    @property
    def output_data_path(self) -> Path:
        return self.common.output_prefix / f'{self.common.data_name}.data'

    @property
    def output_info_path(self) -> Path:
        return self.common.output_prefix / f'{self.common.data_name}.info'
