import re
from typing import override

import numpy as np
import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class CsvdPreprocessor(Preprocessor):
    col_pattern = re.compile(
        r"""
            ^                # Start of string
            (?P<name> .*? )  # Name (non-greedy)
            (?: \( .* \) )?  # Non-greedy match for any comments in parentheses
            / .*             # Slash and English name (ignored)
            $                # End of string
        """,
        re.VERBOSE,
    )

    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    def parse_column_name(self, column: str, previous: str) -> str:
        if col_match := self.col_pattern.match(column):
            name = col_match.group('name')
            if name == '白质占比':
                prefix = previous.split('白质', 1)[0]
                name = f'{prefix}{name}'
            return name
        else:
            raise ValueError(
                f'Column name "{column}" does not match the expected pattern.'
            )

    def rename(self, data: pd.DataFrame) -> pd.DataFrame:
        col_map: dict[str, str] = {}
        previous = ''
        for col in data.columns:
            if col != 'hash_num':
                new_col = self.parse_column_name(col, previous)
                col_map[col] = new_col
                previous = new_col
        data.rename(columns=col_map, inplace=True)
        return data

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config.final:
            data = self.apply_columns(data)
        else:
            # Drop unnecessary columns
            drop_columns = ['检查日期', '性别', '年龄', 'hash_num']
            data = data.drop(columns=drop_columns)
            # Drop columns with (more than) 50% NaN or zero values
            # TODO: Zeros may be valid values and NaNs may be zeros in some columns
            threshold = len(data) * 0.5
            columns = data.columns[
                data.apply(lambda col: col[~col.isin([0, np.nan])].count() >= threshold)
            ]
            data = data[columns]
            # Drop rows with less than 80% non-NaN values in the remaining columns
            threshold = int(len(data.columns) * 0.8)
            data = data.dropna(thresh=threshold)
            # Drop rows with NaN in specific columns
            keywords = ['额叶', '顶叶', '颞叶', '基底节区', '半卵圆中心区']
            columns = [
                col
                for col in data.columns
                if any(keyword in col for keyword in keywords)
            ]
            data = data.dropna(subset=columns)
            # Drop columns having any NaN values
            data = data.dropna(axis=1)
            self.store_columns(data)
        return data

    def binarize(self, data: pd.DataFrame) -> pd.DataFrame:
        binarize_th = 6
        for col in data.columns:
            try:
                if data[col].nunique(dropna=True) <= binarize_th:
                    data = self.sum_columns(data, [col], col)
            except ValueError:
                continue
        return data

    @override
    def get_data(self) -> pd.DataFrame:
        # Set "/" as NaN
        data = self.read_data()

        # Parse column names
        data = self.rename(data)

        # Filter columns and rows
        data = self.filter(data)

        # Binarize certain columns
        data = self.binarize(data)

        return data
