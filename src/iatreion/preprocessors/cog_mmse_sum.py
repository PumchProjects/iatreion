from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class MmseSumPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        # 时间定向
        time_columns = ['星期几', '几号', '几月', '什么季节', '哪一年']
        data = self.sum_columns(data, time_columns, 'MMSE_时间定向')
        # 地点定向
        place_columns = ['省市', '区县', '街道或乡', '什么地方', '几层楼']
        data = self.sum_columns(data, place_columns, 'MMSE_地点定向')
        # 瞬时记忆
        ins_mem_columns = ['皮球', '国旗', '树木']
        data = self.sum_columns(data, ins_mem_columns, 'MMSE_瞬时记忆')
        # 计算功能
        cal_columns = ['减7_1st', '减7_2nd', '减7_3rd', '减7_4th', '减7_5th']
        data = self.sum_columns(data, cal_columns, 'MMSE_计算功能')
        # 延迟记忆
        delay_mem_columns = ['回忆_皮球', '回忆_国旗', '回忆_树木']
        data = self.sum_columns(data, delay_mem_columns, 'MMSE_延迟记忆')
        # 执行功能
        exec_func_columns = ['右手拿纸', '两手对折', '放在左腿上']
        data = self.sum_columns(data, exec_func_columns, 'MMSE_执行功能')
        # 语言功能
        lang_func_columns = [
            '手表',
            '铅笔',
            '四十四只石狮子',
            '请闭上您的眼睛',
            '书写能力',
        ]
        data = self.sum_columns(data, lang_func_columns, 'MMSE_语言功能')
        # 视空间功能
        vis_spa_func_columns = ['结构能力']
        data = self.sum_columns(data, vis_spa_func_columns, 'MMSE_视空间功能')
        # 总分
        data['MMSE_总分'] = data['mmse_selfcalc']
        drop_columns = [col for col in data.columns if not col.startswith('MMSE_')]
        data = self.drop_columns(data, drop_columns)
        return data
