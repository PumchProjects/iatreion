from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class BasicPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()

        data = self.get_basic_info(data, ['性别', '利手', '教育年限'])
        data, real_ages = self.calc_ages(data, '测试日期')

        return pd.DataFrame(
            {
                '性别男': (data['性别'] == '男').astype('Int8'),
                '性别女': (data['性别'] == '女').astype('Int8'),
                '年龄': real_ages,
                '左利手': (data['利手'] == '左利手').astype('Int8'),
                '右利手': (data['利手'] == '右利手').astype('Int8'),
                '双利手': (data['利手'] == '双利手').astype('Int8'),
                '教育年限': data['教育年限'].astype('Float64'),
            }
        )
