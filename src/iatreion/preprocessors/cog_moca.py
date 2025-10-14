from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class MocaPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        recall_columns = ['moca_huiyi']
        data['MoCA_分类提示'] = (
            data['分类提示']
            .div(5 - data[recall_columns[0]].fillna(0), fill_value=0)
            .astype(float)
            .fillna(1)
            .astype('Float64')
        )
        data.loc[data['MoCA_分类提示'] > 1, 'MoCA_分类提示'] = pd.NA
        data = self.sum_columns(data, recall_columns, 'MoCA_回忆')
        data = pd.concat(
            [
                data.loc[:, '连接图形':'骆驼'],  # type: ignore
                data.loc[:, '顺背.2.1.8.5.4':'手表..尺子'],  # type: ignore
                data.loc[:, '日期':'moca_selfcalc'],  # type: ignore
                data.loc[:, 'MoCA_分类提示':],  # type: ignore
            ],
            axis=1,
        )
        return data
