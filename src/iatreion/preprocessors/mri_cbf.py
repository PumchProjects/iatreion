from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class CbfPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        data = data.drop(columns=['date', 'hash_num'])
        # TODO: More specific missing value handling
        # TODO: (directly drop rows with missing values?)
        for column in list(data.columns[data.isnull().sum() > 0]):
            # print(f'Column {column} has missing values.')
            mean_val = data[column].mean()
            data[column] = data[column].fillna(mean_val)
        return data
