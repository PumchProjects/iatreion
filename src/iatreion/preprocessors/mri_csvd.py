import re
from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

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

    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)
        self.is_manual = name.endswith('-manual')

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
        # Drop unnecessary columns
        drop_columns = ['检查日期', '性别', '年龄']
        data = self.drop_columns(data, drop_columns, ['hash_num'])
        if not self.config._final:
            # Drop columns with less than 80% non-NaN values
            threshold = int(len(data) * 0.8)
            data = data.dropna(axis=1, thresh=threshold)
            # Drop rows having any NaN values implicitly
        return data

    def filter_manual(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config._final:
            # Drop columns with less than 85% non-NaN values (including unnamed columns)
            threshold = int(len(data) * 0.85)
            data = data.dropna(axis=1, thresh=threshold)
            # Drop rows having any NaN values implicitly
        return data

    @override
    def get_data(self) -> pd.DataFrame:
        # Set "/" as NaN
        data = self.read_data()

        if self.is_manual:
            # Filter columns and rows
            data = self.filter_manual(data)
        else:
            # Parse column names
            data = self.rename(data)
            # Filter columns and rows
            data = self.filter(data)

        return data
