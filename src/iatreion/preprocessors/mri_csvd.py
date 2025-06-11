from typing import override

import numpy as np
import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class CsvdPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        # Set "/" as NaN
        data = pd.read_excel(self.config.data_path, index_col='serial_num', na_values=['/'])
        # Drop unnecessary columns
        drop_columns = ['检查日期/Study.date', '性别/Sex', '年龄/Age', 'hash_num']
        data = data.drop(columns=drop_columns)
        # Drop columns with (more than) 50% NaN or zero values
        threshold = len(data) * 0.5
        columns = data.columns[data.apply(lambda col: col[~col.isin([0, np.nan])].count() >= threshold)]
        data = data[columns]
        # Drop rows with less than 80% non-NaN values in the remaining columns
        threshold = len(data.columns) * 0.8
        data = data.dropna(thresh=threshold)
        # Drop rows with NaN in specific columns
        keywords = ['额叶', '顶叶', '颞叶', '基底节区', '半卵圆中心区']
        columns = [col for col in data.columns if any(keyword in col for keyword in keywords)]
        data = data.dropna(subset=columns)
        # Drop columns having any NaN values
        data = data.dropna(axis=1)
        return data
