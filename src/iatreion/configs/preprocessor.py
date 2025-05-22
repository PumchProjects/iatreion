from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import Directory

from .dataset import DataName, DatasetConfig

data_name_mapping: dict[DataName, str] = {
    'cbf': '核磁_cbf.xlsx',
    'snp': '基因_snp.csv',
}


@Parameter(name='*')
@dataclass(kw_only=True)
class PreprocessorConfig:
    # HACK: Cyclopts does not support dataclass inheritance yet
    # HACK: The help texts of the parameters are not shown in the help message
    # HACK: Below is the current workaround
    # HACK: Alternative workaround: use Parameter(help=...)
    # TODO: Read Cyclopts documentation to find a better way
    dataset: DatasetConfig

    output_prefix: Annotated[Directory, Parameter(name=['--output', '-o'])]
    'Prefix of the output files.'

    # TODO: Add more parameters for the preprocessor, e.g. filling missing values

    def __post_init__(self) -> None:
        self.output_prefix.mkdir(parents=True, exist_ok=True)

    @property
    def group_data_path(self) -> Path:
        return self.dataset.prefix / '患者及分组加密对应表.xlsx'

    @property
    def data_path(self) -> Path:
        return self.dataset.prefix / data_name_mapping[self.dataset.name]

    @property
    def output_data_path(self) -> Path:
        return self.output_prefix / f'{self.dataset.name}.data'

    @property
    def output_info_path(self) -> Path:
        return self.output_prefix / f'{self.dataset.name}.info'
