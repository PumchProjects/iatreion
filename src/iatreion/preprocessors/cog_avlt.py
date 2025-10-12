from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class AvltPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = data.loc[:, ['AVLT.1', 'AVLT.I', 'AVLT.4', 'AVLT.5', 'AVLT.T']]
        return data
