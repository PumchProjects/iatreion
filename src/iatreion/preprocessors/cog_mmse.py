from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class MmsePreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        part_1 = data.loc[:, '星期几':'树木']  # type: ignore
        part_2 = data.loc[:, '减7_1st':'mmse_selfcalc']  # type: ignore
        data = pd.concat([part_1, part_2], axis=1).dropna()
        return data
