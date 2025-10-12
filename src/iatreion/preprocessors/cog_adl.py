from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class AdlPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, is_sum: bool = False) -> None:
        super().__init__(config)
        self.is_sum = is_sum

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        if not self.is_sum:
            data['其它'] = data['其它'].fillna(0)
            for col in data.loc[:, '自己搭公共汽车':'独自在家'].columns:  # type: ignore
                data = self.sum_columns(data, [col], f'ADL_{col}')
        data['ADL_I'] = data['iadl']
        data['ADL_B'] = data['badl']
        data['ADL_T'] = data['adl']
        drop_columns = [col for col in data.columns if not col.startswith('ADL_')]
        data = self.drop_columns(data, drop_columns)
        return data
