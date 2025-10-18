import re
from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class HadPreprocessor(Preprocessor):
    res_pattern = re.compile(r'^(?P<value>\d+).+$')

    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data.replace(regex=self.res_pattern, value=r'\g<value>', inplace=True)
        if not self.is_sum:
            for col in data.loc[
                :,
                'X.A.1.我感到紧张.或痛苦.':'X.D.14.我能欣赏一本好书或一项好的广播或电视节目',  # type: ignore
            ].columns:
                data = self.sum_columns(data, [col], f'HAD_{col}')
        data['HAD_A'] = pd.to_numeric(data['had_a'], errors='coerce').astype('Int64')
        data['HAD_D'] = pd.to_numeric(data['had_d'], errors='coerce').astype('Int64')
        drop_columns = [col for col in data.columns if not col.startswith('HAD_')]
        data = self.drop_columns(data, drop_columns)
        return data
