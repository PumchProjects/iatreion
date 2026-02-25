import re
from collections.abc import Iterable
from typing import override

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class HistoryPreprocessor(Preprocessor):
    unk_pattern = re.compile(r'^.*(?:undefined|不详|其它).*$')
    val_pattern = re.compile(
        r"""
            ^                       # Start of string
            (?P<value> \d+ )        # Value (number)
            \(                      # Opening parenthesis
            \w+ =                   # Code and equal sign
            (?P<name> .*? )         # Name (non-greedy)
            (?: [(（] .* [）)] )?   # Non-greedy match for any comments in parentheses
            \)                      # Closing parenthesis
            $                       # End of string
        """,
        re.VERBOSE,
    )
    code_pattern = re.compile(
        r"""
            \[                      # Opening bracket
            (?P<code> \w+ )         # Code (characters)
            =                       # Equal sign
            (?P<name> .*? )         # Name (non-greedy)
            (?: [(（] .* [）)] )?   # Non-greedy match for any comments in parentheses
            \]                      # Closing bracket
        """,
        re.VERBOSE,
    )

    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    def select_columns(
        self, data: pd.DataFrame, indices: Iterable[int]
    ) -> pd.DataFrame:
        return data.loc[:, data.columns.intersection([f'V{i}' for i in indices])]

    def convert_object_to_int(
        self, data: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        for col in columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
        return data

    def parse_column_values(
        self, data: pd.DataFrame, column: str
    ) -> tuple[pd.Series, pd.Series]:
        values = data[column]
        names = values.replace(regex=self.val_pattern, value=r'\g<name>')
        values = values.replace(regex=self.val_pattern, value=r'\g<value>').astype(
            'Int64'
        )
        return values, names

    def parse_column_codes(
        self, data: pd.DataFrame, column: str
    ) -> tuple[pd.Series, list[tuple[str, str]]]:
        codes = data[column]
        code_map = self.process_info(dict[str, str], column)
        if not self.config._final:
            for code in codes.unique():
                if not pd.isna(code):
                    for code_match in self.code_pattern.finditer(code):
                        code = code_match.group('code')
                        name = code_match.group('name')
                        code_map[code] = name
        codes.replace(regex=self.code_pattern, value=r'\g<code>', inplace=True)
        code_list = sorted(code_map.items())
        return codes, code_list

    def process_single_choice(
        self, data: pd.DataFrame, column: str, *, ordered: bool = True
    ) -> pd.DataFrame:
        values, names = self.parse_column_values(data, column)
        data[column] = values if ordered else names
        return data

    def binarize_multiple_choice(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        codes, code_list = self.parse_column_codes(data, column)
        data = data.drop(columns=[column])
        for code, name in code_list:
            data[f'{column} = {name}'] = codes.str.contains(code).astype('Int8')
        return data

    def process_data(
        self,
        data: pd.DataFrame,
        threshold: float,
        unordered_columns: list[str],
        multiple_choice_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        # Process each column
        columns = data.columns.to_list()
        multiple_choice_columns = multiple_choice_columns or []
        for col in columns:
            if col in multiple_choice_columns:
                data = self.binarize_multiple_choice(data, col)
            elif col in unordered_columns:
                data = self.process_single_choice(data, col, ordered=False)
            elif is_string_dtype(data[col]):
                data = self.process_single_choice(data, col)
            # Keep continuous columns as is

        if not self.config._final:
            # Drop columns with less than `threshold` non-NaN values
            thresh = int(len(data) * threshold)
            data = data.dropna(axis=1, thresh=thresh)
            # Drop rows having any NaN values implicitly

        return data

    def get_life_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.select_columns(data, range(22, 125))

        # Convert object dtype columns to Int64
        object_columns = ['V62', 'V63', 'V123']
        data = self.convert_object_to_int(data, object_columns)

        threshold = 0.88
        unordered_columns = ['V60', 'V68', 'V81', 'V84', 'V115', 'V116', 'V120']
        multiple_choice_columns = [
            'V22',
            'V33',
            'V41',
            'V54',
            'V70',
            'V74',
            'V75',
            'V86',
            'V90',
            'V91',
            'V97',
            'V101',
            'V105',
        ]
        data = self.process_data(
            data, threshold, unordered_columns, multiple_choice_columns
        )

        return data

    def get_diet_and_medication_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.select_columns(data, range(125, 171))

        threshold = 0.89
        unordered_columns = [f'V{i}' for i in range(136, 171)]
        data = self.process_data(data, threshold, unordered_columns)

        return data

    def get_family_history_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.select_columns(data, range(172, 218))

        # Fit the val_pattern
        data.replace('4(5>3人)', '4(5=超过3人)', inplace=True)

        threshold = 0.86
        unordered_columns = ['V173', 'V174', 'V176']
        data = self.process_data(data, threshold, unordered_columns)

        return data

    def get_medical_history_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.select_columns(data, range(218, 312))

        # Reorder some values
        data.replace(
            {'V240': {'0(1=无)': '1(2=无)', '1(2=减少)': '0(1=减少)'}}, inplace=True
        )

        threshold = 0.9
        unordered_columns = ['V238', 'V279', 'V280', 'V297']
        multiple_choice_columns = ['V293']
        data = self.process_data(
            data, threshold, unordered_columns, multiple_choice_columns
        )

        return data

    def get_symptom_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.select_columns(data, [14] + list(range(312, 394)))

        # Convert V14 to Float64
        data['V14'] = pd.to_numeric(data['V14'], errors='coerce').astype('Float64')

        # Fill data for normal people
        fill_values = {col: '0(a=无)' for col in data.columns}
        fill_values['V312'] = '4(n=无)'
        data.replace('无', fill_values, inplace=True)

        # Reorder some values
        replace_pattern = {
            '0(a=无)': '1(b=无)',
            '1(b=下降)': '0(a=下降)',
            '1(b=减少)': '0(a=减少)',
        }
        replace_columns = ['V373', 'V377', 'V378']
        data.replace({col: replace_pattern for col in replace_columns}, inplace=True)

        threshold = 0.0
        unordered_columns = ['V312']
        multiple_choice_columns = ['V379', 'V380', 'V392', 'V393']
        data = self.process_data(
            data, threshold, unordered_columns, multiple_choice_columns
        )

        return data

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()

        # Rename columns
        data.rename(columns=self.get_name_to_stem_callback(), inplace=True)

        # Set "undefined" as NaN
        data.replace(regex=self.unk_pattern, value=np.nan, inplace=True)

        match self.name:
            case 'life':
                data = self.get_life_data(data)
            case 'diet-medication':
                data = self.get_diet_and_medication_data(data)
            case 'family-history':
                data = self.get_family_history_data(data)
            case 'medical-history':
                data = self.get_medical_history_data(data)
            case 'symptom':
                data = self.get_symptom_data(data)
            case _:
                raise ValueError(f'Unknown part: {self.name}')

        return data
