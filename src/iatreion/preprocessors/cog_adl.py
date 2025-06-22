from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class AdlPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        data['其它'] = data['其它'].fillna(0)
        for col in data.loc[:, '自己搭公共汽车':'独自在家'].columns:
            data = self.sum_columns(data, [col], f'ADL_{col}')
        data['ADL_I'] = data['iadl']
        data['ADL_B'] = data['badl']
        data['ADL_T'] = data['adl']
        drop_columns = [col for col in data.columns if not col.startswith('ADL_')]
        data = data.drop(columns=drop_columns).dropna()
        return data
