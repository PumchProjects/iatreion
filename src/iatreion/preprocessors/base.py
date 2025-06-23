from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from iatreion.configs import PreprocessorConfig
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
        return data[['encrypted', 'Ab']]

    @staticmethod
    def sum_columns(
        data: pd.DataFrame, columns: list[str], name: str, dtype: str = 'Int64'
    ) -> pd.DataFrame:
        # skipna=False ensures that NaN will propagate through the sum
        col: pd.Series = data[columns].sum(axis=1, skipna=False).astype(dtype)
        data = data.drop(columns=columns)
        min_value, max_value = col.min(), col.max()
        data[f'{name} = {min_value}'] = (col == min_value).astype(dtype)
        for th in range(min_value + 1, max_value):
            data[f'{name} <= {th}'] = (col <= th).astype(dtype)
            data[f'{name} >= {th}'] = (col >= th).astype(dtype)
        data[f'{name} = {max_value}'] = (col == max_value).astype(dtype)
        return data

    @abstractmethod
    def get_data(self) -> pd.DataFrame: ...

    def get_augmented_vector_name(self, data: pd.DataFrame) -> list[tuple[str, str]]:
        discrete_th = 10
        augmented_vector_name: list[tuple[str, str]] = []
        for name in data.columns:
            try:
                col = data[name].to_numpy()
                unique_values = np.unique(col[~np.isnan(col)])
                if len(unique_values) <= 2:
                    augmented_vector_name.append((name, 'binary'))
                elif len(unique_values) < discrete_th:
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
        for i, (name, type_) in enumerate(augmented_vector_name[:-2]):
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
        data = data.merge(group_names, left_index=True, right_index=True, copy=False)
        # HACK: keep only the first sample of each patient
        data = data[~data.index.duplicated(keep='first')]
        augmented_vector_name = self.get_augmented_vector_name(data)
        logger.info('[bold green]Saving data...', extra={'markup': True})
        self.save_data(data, augmented_vector_name)
