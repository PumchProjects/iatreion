from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class CompositePreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        selected: list[pd.Series | pd.DataFrame] = []
        data.loc[data['接龙A_错误数'] >= 2, '接龙A_A法时间'] = 1000
        data = self.binarize_column(
            data, '单个动作模仿正确数', 6, '单个动作模仿正常', '单个动作模仿异常'
        )
        data = self.binarize_column(
            data, '系列动作模仿正确数', 2, '系列动作模仿正常', '系列动作模仿异常'
        )
        selected += [
            data.loc[:, '动物列名_实际数字'],
            data.loc[:, ['数字符号_正确数', '接龙A_A法时间']],
        ]
        if self.config.dataset.simple:
            data['Rey再认正确'] = (data['Rey再认'] == '正确').astype('Int8')
            selected += [
                data.loc[:, ['临摹总分', '单个动作模仿正常', 'Rey临摹总分']],
                data.loc[:, '系列动作模仿正常'],
                data.loc[:, ['Rey回忆总分', 'Rey再认正确']],
            ]
        else:
            data['Rey再认正确'] = (data['Rey再认'] == '正确').astype('Int8')
            data['Rey再认错误'] = (data['Rey再认'] == '错误').astype('Int8')
            selected += [
                data.loc[
                    :,
                    [
                        '临摹总分',
                        '单个动作模仿正常',
                        '单个动作模仿异常',
                        'Rey临摹总分',
                    ],
                ],
                data.loc[:, ['系列动作模仿正常', '系列动作模仿异常']],
                data.loc[:, ['Rey回忆总分', 'Rey再认正确', 'Rey再认错误']],
            ]
        selected += [
            data.loc[:, '积木_总分'],
            data.loc[:, '联想学习3次测试总分'],
            data.loc[:, '情景记忆总分'],
            data.loc[:, ['AVLT.1', 'AVLT.I', 'AVLT.4', 'AVLT.5', 'AVLT.T']],
            data.loc[:, ['相似性总分', '计算总分']],
        ]
        data = pd.concat(selected, axis=1)
        return data
