from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class MocaPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        recall_columns = ['moca_huiyi']
        data['MoCA_分类提示'] = (
            data['分类提示']
            .div(5 - data[recall_columns[0]].fillna(0), fill_value=0)
            .fillna(1)
        )
        data = data[data['MoCA_分类提示'] <= 1]
        data = self.sum_columns(data, recall_columns, 'MoCA_回忆')
        part_1 = data.loc[:, '连接图形':'骆驼']
        part_2 = data.loc[:, '顺背.2.1.8.5.4':'手表..尺子']
        part_3 = data.loc[:, '日期':'moca_selfcalc']
        part_4 = data.loc[:, 'MoCA_分类提示':]
        data = pd.concat([part_1, part_2, part_3, part_4], axis=1).dropna()
        return data
