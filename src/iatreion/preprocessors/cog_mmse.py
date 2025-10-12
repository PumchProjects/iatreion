from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class MmsePreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = pd.concat(
            [
                data.loc[:, '星期几':'树木'],  # type: ignore
                data.loc[:, '减7_1st':'mmse_selfcalc'],  # type: ignore
            ],
            axis=1,
        )
        return data
