from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import load_dict, logger, save_dict

from .process_info import ProcessInfo


class Preprocessor(ABC):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__()
        self.config = config
        self.process_info_dict_: dict[str, dict[str, Any]] | None = None
        self.process_info_: ProcessInfo | None = None

    @property
    def process_info_dict(self) -> dict[str, dict[str, Any]]:
        if self.process_info_dict_ is None:
            self.process_info_dict_ = load_dict(self.config.process_info_path)
        return self.process_info_dict_

    @property
    def process_info(self) -> ProcessInfo:
        if self.process_info_ is None:
            name = self.config.dataset.name
            if self.config.final:
                if name not in self.process_info_dict:
                    raise IatreionException(
                        'No processing info found for "$dataset"',
                        dataset=name,
                    )
                else:
                    info = self.process_info_dict[name]
                    self.process_info_ = ProcessInfo(name, info, final=True)
            else:
                self.process_info_ = ProcessInfo(name, final=False)
        return self.process_info_

    def store_columns(self, data: pd.DataFrame) -> None:
        self.process_info['columns'] = data.columns.tolist()

    def apply_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        # Keep only the columns in the processing info
        return data[self.process_info['columns']].dropna()

    def save_process_info(self) -> None:
        if not self.config.final and self.process_info_ is not None:
            info = self.process_info_.attributes
            self.process_info_dict[self.config.dataset.name] = info
            save_dict(self.process_info_dict, self.config.process_info_path)

    def get_group_names(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.group_data_path, index_col='serial_num')
        data.rename(
            columns={
                'group_encrypted': 'encrypted',
                'group_Ab': 'Ab',
            },
            inplace=True,
        )
        return data[self.config.dataset.group_columns]

    def get_birth_dates(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self.config.final:
            birth_dates = pd.to_datetime(data['date of birth'], utc=True)
            return data, birth_dates
        else:
            birth_data = pd.read_excel(
                self.config.birth_data_path, index_col='serial_num'
            )
            birth_dates = pd.to_datetime(birth_data['实际出生日期'])
            data = data.merge(birth_dates, left_index=True, right_index=True)
            return data, data['实际出生日期']

    def sum_columns(
        self, data: pd.DataFrame, columns: list[str], name: str
    ) -> pd.DataFrame:
        # skipna=False ensures that NaN will propagate through the sum
        col: pd.Series = data[columns].sum(axis=1, skipna=False).astype('Int64')
        data = data.drop(columns=columns)
        if self.config.dataset.simple:
            data[name] = col
        else:
            min_value, max_value = col.min(), col.max()
            data[f'{name} <= {min_value}'] = (col <= min_value).astype('Int8')
            for th in range(min_value + 1, max_value):
                data[f'{name} <= {th}'] = (col <= th).astype('Int8')
                data[f'{name} >= {th}'] = (col >= th).astype('Int8')
            data[f'{name} >= {max_value}'] = (col >= max_value).astype('Int8')
        return data

    def binarize_column(
        self,
        data: pd.DataFrame,
        column: str,
        threshold: int,
        ge_name: str,
        lt_name: str,
        ge_main: bool = True,
    ) -> pd.DataFrame:
        col: pd.Series = (data[column] >= threshold).astype('Int8')
        data = data.drop(columns=[column])
        if self.config.dataset.simple:
            data[ge_name if ge_main else lt_name] = col
        else:
            data[ge_name] = (col == 1).astype('Int8')
            data[lt_name] = (col == 0).astype('Int8')
        return data

    def read_data(self) -> pd.DataFrame:
        data = pd.read_excel(
            self.config.data_path,
            index_col='ID' if self.config.final else 'serial_num',
            na_values=['/', '#NUM!'],
            dtype_backend='numpy_nullable',
        )
        return data

    @abstractmethod
    def get_data(self) -> pd.DataFrame: ...

    @staticmethod
    def deduplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
        # HACK: Keep only the last sample of each patient
        data = data[~data.index.duplicated(keep='last')]
        return data

    def get_data_outer(self) -> pd.DataFrame:
        data = self.deduplicate_rows(self.get_data())
        self.save_process_info()
        return data

    def get_child_data(self, name: DataName, child: Self) -> pd.DataFrame:
        original_name = self.config.dataset.name
        self.config.dataset.name = name
        data = child.get_data_outer()
        self.config.dataset.name = original_name
        return data

    @staticmethod
    def remove_useless_columns(data: pd.DataFrame) -> pd.DataFrame:
        nunique = data.nunique(dropna=False)
        columns = nunique[nunique <= 1].index
        if not columns.empty:
            logger.warning(
                f'[bold yellow]Removing useless columns:[/] {", ".join(columns)}',
                extra={'markup': True},
            )
            data = data.drop(columns=columns)
        return data

    def get_augmented_vector_name(self, data: pd.DataFrame) -> list[tuple[str, str]]:
        discrete_th = 4
        augmented_vector_name: list[tuple[str, str]] = []
        for name in data.columns:
            try:
                col = data[name].to_numpy()
                unique_values = np.unique(col[~np.isnan(col)])
                if len(unique_values) <= 2:
                    augmented_vector_name.append((name, 'binary'))
                elif (
                    len(unique_values) < discrete_th and not self.config.dataset.simple
                ):
                    augmented_vector_name.append((name, 'discrete'))
                else:
                    augmented_vector_name.append((name, 'continuous'))
            except TypeError:
                augmented_vector_name.append((name, 'discrete'))
        return augmented_vector_name

    def save_data(
        self, data: pd.DataFrame, augmented_vector_name: list[tuple[str, str]]
    ) -> None:
        feature_names = [f'{pair[0]} {pair[1]}\n' for pair in augmented_vector_name]
        with self.config.output_info_path.open('w', encoding='utf-8') as f:
            f.writelines(feature_names)
        fmap: list[str] = []
        for i, (name_, type_) in enumerate(
            augmented_vector_name[: -len(self.config.dataset.group_columns)]
        ):
            name = name_.replace(' ', self.config.dataset.place_holder)
            match type_:
                case 'binary':
                    fmap.append(f'{i}\t{name}\ti\n')
                case 'continuous':
                    fmap.append(f'{i}\t{name}\tq\n')
                case _:
                    raise ValueError(f'Unsupported type `{type_}` for `{name_}`')
        with self.config.output_fmap_path.open('w', encoding='utf-8') as f:
            f.writelines(fmap)
        with self.config.output_data_path.open('w', encoding='utf-8') as f:
            raw = data.to_string(header=False, index=False, index_names=False).split(
                '\n'
            )
            f.write('\n'.join([','.join(element.split()) for element in raw]))

    def process(self) -> None:
        group_names = self.get_group_names()
        data = self.get_data_outer()
        data = data.merge(group_names, left_index=True, right_index=True)
        # HACK: Just in case there are duplicate IDs in the group file
        data = self.deduplicate_rows(data)
        data = self.remove_useless_columns(data)
        augmented_vector_name = self.get_augmented_vector_name(data)
        logger.info('[bold green]Saving data...', extra={'markup': True})
        self.save_data(data, augmented_vector_name)


type NamedPreprocessor = tuple[DataName, Preprocessor]
