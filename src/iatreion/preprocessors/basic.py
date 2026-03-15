from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class BasicPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, name: DataName, age: bool = True
    ) -> None:
        super().__init__(config, name)
        self.age = age

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()

        selected = ['性别', '利手', '教育年限']
        data = self.get_basic_info(data, selected)
        data, real_ages = self.calc_ages(data, '测试日期')

        for col in ['性别', '利手']:
            if self.config._final:
                categories = self.process_info(list[str], col, 'categories')
                data[col] = pd.Categorical(data[col], categories=categories)
            else:
                data[col] = data[col].astype('category')
                self.process_info[col, 'categories'] = data[col].cat.categories.tolist()

        if self.age:
            selected.append('年龄')
            data['年龄'] = real_ages

        return data[selected]
