from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig
from iatreion.utils import logger


class Preprocessor(ABC):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__()
        self.config = config

    def get_group_names(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.group_data_path, index_col='serial_num')
        data.rename(
            columns={
                'group_encrypted': 'encrypted',
                'group_Ab': 'Ab',
            },
            inplace=True,
        )
        return data[['encrypted', 'Ab', 'A_type', 'A_type2']]

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
        min_value, max_value = col.min(), col.max()
        if self.config.dataset.simple:
            data[name] = col
        else:
            data[f'{name} = {min_value}'] = (col == min_value).astype('Int64')
            for th in range(min_value + 1, max_value):
                data[f'{name} <= {th}'] = (col <= th).astype('Int64')
                data[f'{name} >= {th}'] = (col >= th).astype('Int64')
            data[f'{name} = {max_value}'] = (col == max_value).astype('Int64')
        return data

    def binarize_column(
        self,
        data: pd.DataFrame,
        column: str,
        threshold: int,
        ge_name: str | None = None,
        lt_name: str | None = None,
    ) -> pd.DataFrame:
        col: pd.Series = (data[column] >= threshold).astype('Int64')
        col[data[column].isnull()] = np.nan
        data = data.drop(columns=[column])
        if ge_name is not None:
            name = ge_name
        else:
            assert lt_name is not None, (
                'At least one of ge_name or lt_name must be provided'
            )
            name = lt_name
            col = 1 - col
        if self.config.dataset.simple:
            data[name] = col
        else:
            data[f'{name} = 0'] = (col == 0).astype('Int64')
            data[f'{name} = 1'] = (col == 1).astype('Int64')
        return data

    def read_data(self) -> pd.DataFrame:
        data = pd.read_excel(
            self.config.data_path,
            index_col='ID' if self.config.final else 'serial_num',
            na_values=['/'],
        )
        return data

    @abstractmethod
    def get_data(self) -> pd.DataFrame: ...

    @staticmethod
    def deduplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
        # HACK: Keep only the first sample of each patient
        data = data[~data.index.duplicated(keep='first')]
        return data

    def get_child_data(self, name: DataName, child: Self) -> pd.DataFrame:
        original_name = self.config.dataset.name
        self.config.dataset.name = name
        data = self.deduplicate_rows(child.get_data())
        self.config.dataset.name = original_name
        return data

    @staticmethod
    def remove_useless_columns(data: pd.DataFrame) -> pd.DataFrame:
        nunique = data.nunique(dropna=False)
        columns = nunique[nunique <= 1].index
        if not columns.empty:
            logger.warning(
                f'[bold yellow]Removing useless columns: {", ".join(columns)}',
                extra={'markup': True},
            )
            data = data.drop(columns=columns)
        return data

    def get_augmented_vector_name(self, data: pd.DataFrame) -> list[tuple[str, str]]:
        discrete_th = 10
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
        for i, (name, type_) in enumerate(augmented_vector_name[:-4]):
            name = name.replace(' ', self.config.dataset.place_holder)
            match type_:
                case 'binary':
                    fmap.append(f'{i}\t{name}\ti\n')
                case 'continuous':
                    fmap.append(f'{i}\t{name}\tq\n')
                case _:
                    raise ValueError(f'Unsupported type: {type_}')
        with self.config.output_fmap_path.open('w', encoding='utf-8') as f:
            f.writelines(fmap)
        with self.config.output_data_path.open('w', encoding='utf-8') as f:
            raw = data.to_string(header=False, index=False, index_names=False).split(
                '\n'
            )
            f.write('\n'.join([','.join(element.split()) for element in raw]))

    def process(self) -> None:
        group_names = self.get_group_names()
        data = self.get_data()
        data = data.merge(group_names, left_index=True, right_index=True)
        data = self.deduplicate_rows(data)
        data = self.remove_useless_columns(data)
        augmented_vector_name = self.get_augmented_vector_name(data)
        logger.info('[bold green]Saving data...', extra={'markup': True})
        self.save_data(data, augmented_vector_name)


type NamedPreprocessor = tuple[DataName, Preprocessor]
