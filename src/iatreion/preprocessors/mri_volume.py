from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class VolumePreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        columns = [col for col in data.columns if col.endswith('_Z')]
        data = data[columns].dropna(axis=1, how='all').dropna()
        return data
