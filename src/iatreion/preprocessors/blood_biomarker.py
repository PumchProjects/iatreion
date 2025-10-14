from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class BiomarkerPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = self.drop_columns(data, None, ['hash_num'])
        return data
