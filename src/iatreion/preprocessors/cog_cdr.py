from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class CdrPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        drop_columns = ['性别', '出生年月日', '受教育年限', '填表日期', 'hash_num']
        data = data.drop(columns=drop_columns).dropna()
        return data
