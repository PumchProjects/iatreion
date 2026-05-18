from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from cyclopts import Parameter
from cyclopts.types import ExistingFile

from iatreion.exceptions import IatreionException
from iatreion.utils import load_dict, logger, save_dict

from .dataset import DataName, DatasetConfig

data_indices_mapping: dict[str, list[str]] = {
    'history': [],
    'cdr': ['填表日期'],
    'screen': ['测试日期'],
    'composite': ['填表日期'],
    'biomarker': ['采样时间'],
    'cbf': ['date'],
    'csvd': ['检查日期/Study.date'],
    'csvd-manual': [],
    'volume': ['MRI_time'],
    'volume-new': ['检查日期/Study date'],
    'volume-adni': ['检查日期/Study date'],
    'snp': [],
}

data_stem_mapping: dict[str, str] = {
    'history': r'V\d+',
}

name_data_mapping: dict[DataName, str] = {
    'basic-noage': 'screen',
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
    'mmse-sum-pct': 'screen',
    'moca': 'screen',
    'moca-sum': 'screen',
    'moca-sum-pct': 'screen',
    'adl': 'screen',
    'adl-sum': 'screen',
    'had': 'screen',
    'had-sum': 'screen',
    's-screen-noage-sum': 'screen',
    's-screen-sum': 'screen',
    's-screen-noage-sum-pct': 'screen',
    's-screen-sum-pct': 'screen',
    'associative-learning': 'composite',
    'episodic-memory': 'composite',
    'avlt': 'composite',
    's-composite-aea': 'composite',
    'composite-bin': 'composite',
    'biomarker': 'biomarker',
    'cbf': 'cbf',
    'csvd': 'csvd',
    'csvd-manual': 'csvd-manual',
    'volume': 'volume',
    'volume-v': 'volume',
    'volume-pct': 'volume',
    'volume-z-v': 'volume',
    'volume-z-pct': 'volume',
    'volume-nz-v': 'volume',
    'volume-nz-pct': 'volume',
    'volume-new-v': 'volume-new',
    'volume-new-pct': 'volume-new',
    'volume-adni-v': 'volume-adni',
    'volume-adni-pct': 'volume-adni',
    'snp': 'snp',
}

sequence_mapping: dict[DataName, list[DataName]] = {
    's-history': [
        'life',
        'diet-medication',
        'family-history',
        'medical-history',
        'symptom',
    ],
    's-screen-noage-sum': ['basic-noage', 'mmse-sum', 'moca-sum', 'adl-sum', 'had-sum'],
    's-screen-sum': ['basic', 'mmse-sum', 'moca-sum', 'adl-sum', 'had-sum'],
    's-screen-noage-sum-pct': [
        'basic-noage',
        'mmse-sum-pct',
        'moca-sum-pct',
        'adl-sum',
        'had-sum',
    ],
    's-screen-sum-pct': ['basic', 'mmse-sum-pct', 'moca-sum-pct', 'adl-sum', 'had-sum'],
    's-composite-aea': ['associative-learning', 'episodic-memory', 'avlt'],
}

valid_data_names = set(data_indices_mapping)


@Parameter(name='*')
@dataclass(kw_only=True)
class PreprocessorConfig:
    dataset: DatasetConfig

    data: Annotated[dict[str, ExistingFile], Parameter(alias='-d')] = field(
        default_factory=dict
    )
    'Input files keyed by raw data name.'

    data_sheets: Annotated[dict[str, str], Parameter(alias='-ds')] = field(
        default_factory=dict
    )
    'Excel sheet names or indices keyed by raw data name. If not set, use sheet 0.'

    group_data: Annotated[ExistingFile | None, Parameter(alias='-gd')] = None
    'Patient group mapping file.'

    basic_data: Annotated[ExistingFile | None, Parameter(alias='-bd')] = None
    'Basic patient information file.'

    vmri: Annotated[ExistingFile | None, Parameter(alias='-v')] = None
    'Path to the Vmri_mean_sd data file.'

    vmri_change: Annotated[ExistingFile | None, Parameter(alias='-vc')] = None
    'Path to the Vmri_mean_sd column name change file.'

    index_name_: Annotated[str | None, Parameter(alias='-in')] = None
    'Index column name in the data files. If not set, use default index name.'

    group_columns_: Annotated[
        list[str] | None, Parameter(alias='-gc', consume_multiple=True)
    ] = None
    'Group columns in the data files. If not set, use default group columns.'

    discrete_threshold: Annotated[int, Parameter(alias='-dt')] = 10
    """Threshold for determining whether a column is discrete. If the number of unique
values in a column is less than or equal to this threshold, it will be considered as
discrete. This is used for determining the encoding method for the column.
"""

    _process_info_path: Path | None = None

    _final: bool = False

    _keep: Literal['first', 'last'] = 'last'

    _data: dict[str, pd.DataFrame] = field(default_factory=dict[str, pd.DataFrame])

    _final_indices: list[pd.DataFrame] = field(default_factory=list[pd.DataFrame])

    _process_info_dict: dict[str, dict[str, Any]] | None = None

    # TODO: Add more parameters for the preprocessor, e.g. filling missing values

    def __post_init__(self) -> None:
        self.dataset.prefix.mkdir(parents=True, exist_ok=True)
        self.validate_input_paths()

    @staticmethod
    def format_names(names: set[str]) -> str:
        return ', '.join(sorted(names))

    def validate_input_paths(self) -> None:
        unknown_data_names = set(self.data) - valid_data_names
        if unknown_data_names:
            raise IatreionException(
                'Unknown data path key(s): $data_names',
                data_names=self.format_names(unknown_data_names),
            )
        unknown_sheet_names = set(self.data_sheets) - valid_data_names
        if unknown_sheet_names:
            raise IatreionException(
                'Unknown data sheet key(s): $data_names',
                data_names=self.format_names(unknown_sheet_names),
            )

    @property
    def index_name(self) -> str:
        if self.index_name_ is not None:
            return self.index_name_
        if self._final:
            raise IatreionException('$index_name must be set', index_name='Index name')
        return 'serial_num'

    @property
    def group_data_path(self) -> Path:
        if self.group_data is None:
            raise IatreionException('$group_data must be set', group_data='Group data')
        return self.group_data

    @property
    def basic_data_path(self) -> Path:
        if self.basic_data is None:
            raise IatreionException('$basic_data must be set', basic_data='Basic data')
        return self.basic_data

    @property
    def vmri_data_path(self) -> Path:
        if self.vmri is None:
            raise IatreionException('$vmri must be set', vmri='VMRI')
        return self.vmri

    @property
    def vmri_change_path(self) -> Path:
        if self.vmri_change is None:
            raise IatreionException(
                '$vmri_change must be set', vmri_change='VMRI change'
            )
        return self.vmri_change

    @staticmethod
    def get_data_name(name: DataName) -> str:
        return name_data_mapping[name]

    def get_data_path(self, data_name: str) -> tuple[Path, int | str]:
        if data_name not in self.data:
            raise IatreionException(
                'Data name "$data_name" not found in data paths.',
                data_name=data_name,
            )
        sheet_name = self.data_sheets.get(data_name)
        if sheet_name is None:
            sheet: int | str = 0
        elif sheet_name.isdecimal():
            sheet = int(sheet_name)
        else:
            sheet = sheet_name
        return self.data[data_name], sheet

    @staticmethod
    def get_indices_names(data_name: str) -> list[str]:
        return data_indices_mapping[data_name]

    @property
    def contains_group_columns(self) -> bool:
        contains = self.group_columns_ is not None
        if self._final and not contains:
            raise IatreionException(
                '$label_name must be set in eval mode', label_name='Label name'
            )
        return contains

    @property
    def group_columns(self) -> list[str]:
        if self.group_columns_ is not None:
            return self.group_columns_
        return ['group_encrypted', 'group_Ab', 'AC to 3', 'AC 60']

    @staticmethod
    def get_stem_pattern(data_name: str) -> str | None:
        return data_stem_mapping.get(data_name)

    @property
    def process_info_path(self) -> Path:
        if self._process_info_path is not None:
            return self._process_info_path
        if self._final:
            raise IatreionException(
                '$process_info must be set', process_info='Processing info'
            )
        return self.dataset.prefix / 'process_info.toml'

    def children_names(self, name: DataName) -> list[DataName]:
        return sequence_mapping.get(name, [])

    @property
    def process_info_dict(self) -> dict[str, dict[str, Any]]:
        if self._process_info_dict is None:
            self._process_info_dict = load_dict(self.process_info_path)
        return self._process_info_dict

    def save_process_info_dict(self) -> None:
        if self._process_info_dict is not None:
            logger.info('[bold green]Saving processing info...', extra={'markup': True})
            save_dict(self._process_info_dict, self.process_info_path)
