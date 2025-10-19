from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, cast

import pandas as pd
from cyclopts import Parameter
from cyclopts.types import ExistingDirectory

from iatreion.exceptions import IatreionException
from iatreion.utils import load_dict, logger, save_dict

from .dataset import DataName, DatasetConfig

data_file_mapping: dict[str, str] = {
    'history': '病史_20250924.xlsx',
    'cdr': '认知_cdr.xlsx',
    'screen': '认知筛查.xlsx',
    'composite': '认知综合.xlsx',
    'biomarker': '血液生物标记物_blood_biomarker.xlsx',
    'cbf': '核磁_cbf.xlsx',
    'csvd': '核磁_csvd_20251008.xlsx',
    'volume': '核磁_volume.xlsx',
    'volume-new': '核磁_volume202510_历次.xlsx',
    'snp': '基因_snp.csv',
    'test-screen': '认证报告_20251016.xlsx@sc',
    'test-volume': '认证报告_20251016.xlsx@mri',
}

data_indices_mapping: dict[str, list[str]] = {
    'history': [],
    'cdr': ['填表日期'],
    'screen': ['测试日期'],
    'composite': ['填表日期'],
    'biomarker': [],
    'cbf': ['date'],
    'csvd': ['检查日期/Study.date'],
    'volume': ['MRI_time'],
    'volume-new': ['检查日期/Study date'],
    'snp': [],
    'test-screen': ['测试日期'],
    'test-volume': ['MRI_time'],
}

data_level_mapping: dict[str, str] = {
    'volume-new': 'level_type',
}

name_data_mapping: dict[DataName, str] = {
    'basic': 'screen',
    'life': 'history',
    'diet-medication': 'history',
    'family-history': 'history',
    'medical-history': 'history',
    'symptom': 'history',
    's-history': 'history',
    'cdr': 'cdr',
    'mmse': 'screen',
    'mmse-sum': 'screen',
    'moca': 'screen',
    'moca-sum': 'screen',
    'adl': 'screen',
    'adl-sum': 'screen',
    'had': 'screen',
    'had-sum': 'screen',
    's-screen-sum': 'screen',
    'associative-learning': 'composite',
    'episodic-memory': 'composite',
    'avlt': 'composite',
    's-composite-aea': 'composite',
    'composite-bin': 'composite',
    'biomarker': 'biomarker',
    'cbf': 'cbf',
    'csvd': 'csvd',
    'volume': 'volume',
    'volume-v': 'volume',
    'volume-pct': 'volume',
    'volume-z-v': 'volume',
    'volume-z-pct': 'volume',
    'volume-nz-v': 'volume',
    'volume-nz-pct': 'volume',
    'volume-new-v': 'volume-new',
    'volume-new-pct': 'volume-new',
    'snp': 'snp',
    'test-mmse-sum': 'test-screen',
    'test-moca-sum': 'test-screen',
    'test-adl-sum': 'test-screen',
    'test-had-sum': 'test-screen',
    'test-s-screen-sum': 'test-screen',
    'test-volume-z-pct': 'test-volume',
}

sequence_mapping: dict[DataName, list[DataName]] = {
    's-history': [
        'life',
        'diet-medication',
        'family-history',
        'medical-history',
        'symptom',
    ],
    's-screen-sum': ['basic', 'mmse-sum', 'moca-sum', 'adl-sum', 'had-sum'],
    's-composite-aea': ['associative-learning', 'episodic-memory', 'avlt'],
    'test-s-screen-sum': [
        'test-mmse-sum',
        'test-moca-sum',
        'test-adl-sum',
        'test-had-sum',
    ],
    'test-s-all': ['test-s-screen-sum', 'test-volume-z-pct'],
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

    input_prefix: Annotated[ExistingDirectory, Parameter(name=['--input', '-i'])]
    'Prefix of the input files.'

    vmri_data_path_: Annotated[Path | None, Parameter(parse=False)] = None

    vmri_change_path_: Annotated[Path | None, Parameter(parse=False)] = None

    data_paths: Annotated[dict[str, Path] | None, Parameter(parse=False)] = None

    process_info_path_: Annotated[Path | None, Parameter(parse=False)] = None

    final: Annotated[bool, Parameter(parse=False)] = False

    debug: Annotated[bool, Parameter(parse=False)] = False

    data: Annotated[dict[str, pd.DataFrame], Parameter(parse=False)] = field(
        default_factory=dict
    )

    final_indices: Annotated[list[pd.DataFrame], Parameter(parse=False)] = field(
        default_factory=list
    )

    process_info_dict_: Annotated[
        dict[str, dict[str, Any]] | None, Parameter(parse=False)
    ] = None

    # TODO: Add more parameters for the preprocessor, e.g. filling missing values

    def __post_init__(self) -> None:
        self.dataset.prefix.mkdir(parents=True, exist_ok=True)

    @property
    def group_data_path(self) -> Path:
        return self.input_prefix / '副本患者及分组加密对应表202510_.xlsx'

    @property
    def basic_data_path(self) -> Path:
        return self.input_prefix / '基本信息202510.xlsx'

    @property
    def vmri_data_path(self) -> Path:
        if self.vmri_data_path_ is not None:
            if not self.vmri_data_path_.is_file():
                raise IatreionException('$vmri file is not found', vmri='VMRI')
            return self.vmri_data_path_
        return self.input_prefix / 'Vmri_mean_sd.xlsx'

    @property
    def vmri_change_path(self) -> Path:
        if self.vmri_change_path_ is not None:
            if not self.vmri_change_path_.is_file():
                raise IatreionException(
                    '$vmri_change file is not found', vmri_change='VMRI change'
                )
            return self.vmri_change_path_
        return self.input_prefix / '表头变化202510.xlsx'

    @staticmethod
    def get_data_name(name: DataName) -> str:
        return name_data_mapping[name]

    def get_data_path(self, data_name: str) -> tuple[Path, int | str]:
        if self.data_paths is not None:
            if data_name not in self.data_paths:
                raise IatreionException(
                    'Data name "$data_name" not found in data paths.',
                    data_name=data_name,
                )
            return self.data_paths[data_name], 0
        file_name = data_file_mapping[data_name].rsplit('@', maxsplit=1)
        sheet_name = file_name[1] if len(file_name) == 2 else 0
        return self.input_prefix / file_name[0], sheet_name

    @staticmethod
    def get_indices_names(data_name: str) -> list[str]:
        return data_indices_mapping[data_name]

    @staticmethod
    def get_level_name(data_name: str) -> str | None:
        return data_level_mapping.get(data_name)

    @property
    def process_info_path(self) -> Path:
        if self.process_info_path_ is not None:
            if not self.process_info_path_.is_file():
                raise IatreionException(
                    '$process_info file is not found', process_info='Processing info'
                )
            return self.process_info_path_
        return self.dataset.prefix / 'process_info.toml'

    @staticmethod
    def get_stem(name: DataName) -> DataName:
        return cast(DataName, name.removeprefix('test-'))

    def children_names(self, name: DataName) -> list[DataName]:
        return sequence_mapping.get(name, [])

    @property
    def process_info_dict(self) -> dict[str, dict[str, Any]]:
        if self.process_info_dict_ is None:
            self.process_info_dict_ = load_dict(self.process_info_path)
        return self.process_info_dict_

    def save_process_info_dict(self) -> None:
        if self.process_info_dict_ is not None:
            logger.info('[bold green]Saving processing info...', extra={'markup': True})
            save_dict(self.process_info_dict_, self.process_info_path)
