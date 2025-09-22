from collections import defaultdict
from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class SnpPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        # TODO: Use self.read_data()
        column_dtypes = defaultdict(lambda: 'Int8')
        column_dtypes['X'] = int
        column_dtypes['Row.names'] = str
        data = pd.read_csv(
            self.config.data_path, index_col='Row.names', dtype=column_dtypes
        )
        data = data.drop(columns=['X']).T
        return data
