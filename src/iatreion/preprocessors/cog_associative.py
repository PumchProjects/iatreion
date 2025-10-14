from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class AssociativeLearningPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = data.loc[:, ['联想学习第1次测试', '联想学习3次测试总分']]
        return data
