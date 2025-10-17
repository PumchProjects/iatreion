from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class SnpPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        # TODO: Use self.read_data()
        data = pd.read_csv(
            self.config.get_data_path(self.data_name)[0], index_col='Row.names'
        )
        data = data.drop(columns=['X']).astype('Int8').T
        return data
