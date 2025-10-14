from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class CdrPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        drop_columns = ['性别', '出生年月日', '受教育年限', '填表日期']
        data = self.drop_columns(data, drop_columns, ['hash_num'])
        return data
