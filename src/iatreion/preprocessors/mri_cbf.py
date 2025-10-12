from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class CbfPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = self.drop_columns(data, ['date'], ['hash_num'])
        return data
