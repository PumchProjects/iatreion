from typing import Literal, override

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


class VolumeAveragePreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, feature: Literal['v', 'pct']
    ) -> None:
        super().__init__(config)
        self.feature = feature

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        ai_columns = [col for col in data.columns if col.endswith('_Asymmetry_index')]
        columns = [col for col in data.columns if col.endswith(f'_{self.feature}_Z')]
        c_columns = [
            col
            for col in columns
            if not col.endswith(f'_L_{self.feature}_Z')
            and not col.endswith(f'_R_{self.feature}_Z')
        ]
        lr_columns = [
            col.removesuffix(f'_L_{self.feature}_Z')
            for col in columns
            if col.endswith(f'_L_{self.feature}_Z')
        ]
        a_columns = []
        for col in lr_columns:
            a_columns.append(f'{col}_A_{self.feature}_Z')
            data[f'{col}_A_{self.feature}_Z'] = (
                data[f'{col}_L_{self.feature}_Z'] + data[f'{col}_R_{self.feature}_Z']
            ) * 0.5
        data = (
            data[c_columns + a_columns + ai_columns].dropna(axis=1, how='all').dropna()
        )
        return data
