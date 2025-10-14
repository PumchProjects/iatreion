from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class MocaSumPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        # 视空间功能
        vis_spa_func_columns = ['连接图形', '复制立方体']
        data = self.sum_columns(data, vis_spa_func_columns, 'MoCA_视空间功能')
        # 画钟
        clock_columns = ['轮廓', '数字', '指针']
        data = self.sum_columns(data, clock_columns, 'MoCA_画钟')
        # 瞬时记忆
        ins_mem_columns = ['狮子', '水牛', '骆驼']
        data = self.sum_columns(data, ins_mem_columns, 'MoCA_瞬时记忆')
        # 注意能力
        attention_columns = ['顺背.2.1.8.5.4', '倒背.7.4.2', '读字母']
        data = self.sum_columns(data, attention_columns, 'MoCA_注意能力')
        # 计算能力
        cal_columns = ['X93', 'X86', 'X79', 'X72', 'X65']
        data = self.sum_columns(data, cal_columns, 'MoCA_计算能力')
        # 语言功能
        lang_func_columns = [
            '他出去以后还没有回来',
            '当他回到家的时候.发现屋子里坐满了朋友',
            '在1分钟内尽可能多地说出动物的名字',
        ]
        data = self.sum_columns(data, lang_func_columns, 'MoCA_语言功能')
        # 概念推理
        concept_columns = ['火车..自行车', '手表..尺子']
        data = self.sum_columns(data, concept_columns, 'MoCA_概念推理')
        # 回忆
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
        # 时间定向
        time_columns = ['日期', '月份', '年代', '星期']
        data = self.sum_columns(data, time_columns, 'MoCA_时间定向')
        # 地点定向
        place_columns = ['地点', '城市']
        data = self.sum_columns(data, place_columns, 'MoCA_地点定向')
        # 总分
        data['MoCA_总分'] = data['moca_selfcalc']
        drop_columns = [col for col in data.columns if not col.startswith('MoCA_')]
        data = self.drop_columns(data, drop_columns)
        return data
