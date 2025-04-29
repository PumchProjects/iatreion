from abc import abstractmethod

import numpy as np
import pandas as pd
from autoregistry import Registry

from iatreion.configs import PreprocessorConfig


class Preprocessor(Registry, suffix='Preprocessor'):
    def __init__(self, config: PreprocessorConfig, data_full_name: str) -> None:
        super().__init__()
        self.config = config
        data_name = config.data_name
        self.data_path = self.config.data_prefix / data_full_name
        self.output_data_path = self.config.output_prefix / f'{data_name}.data'
        self.output_info_path = self.config.output_prefix / f'{data_name}.info'
        self.config.output_prefix.mkdir(parents=True, exist_ok=True)

    def get_group_names(self) -> pd.Series:
        data_path = self.config.data_prefix / '患者及分组加密对应表.xlsx'
        data = pd.read_excel(data_path, index_col='serial_num')
        group_Ab = '1' in self.config.group or '2' in self.config.group
        group_names = data['group_Ab'] if group_Ab else data['group_encrypted']
        group_names = group_names[group_names.isin(list(self.config.group))]
        return group_names

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
        with self.output_info_path.open('w', encoding='utf-8') as f:
            f.writelines(feature_names)
            f.write('y discrete\nLABEL_POS -1')
        with self.output_data_path.open('w', encoding='utf-8') as f:
            raw = data.to_string(header=False, index=False, index_names=False).split(
                '\n'
            )
            f.write('\n'.join([','.join(element.split()) for element in raw]))

    def process(self) -> None:
        group_names = self.get_group_names()
        data = self.get_data()
        augmented_vector_name = self.get_augmented_vector_name(data)
        data = data.merge(group_names, left_index=True, right_index=True, copy=False)
        self.save_data(data, augmented_vector_name)
