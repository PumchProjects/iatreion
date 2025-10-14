from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class AvltPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = data.loc[:, ['AVLT.1', 'AVLT.I', 'AVLT.4', 'AVLT.5', 'AVLT.T']]
        return data
