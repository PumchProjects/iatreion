from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class BiomarkerPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        data = data.drop(columns=['hash_num']).dropna()
        return data
