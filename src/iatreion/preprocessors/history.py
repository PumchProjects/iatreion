import re
from typing import override

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class HistoryPreprocessor(Preprocessor):
    unk_pattern = re.compile(r'^.*(?:undefined|不详).*$')
    col_pattern = re.compile(
        r"""
            ^                       # Start of string
            V (?P<index> \d+ ) _    # V, index, and underscore
            (?: .*- (?=[IⅢ起]) )?   # Non-greedy match for comments followed by a dash
            (?P<stage> [IVⅢ]* )     # Stage (I, II, III, IV, Ⅲ, or empty)
            (?P<name> .*? )         # Name (non-greedy)
            (?: [(（] .* [）)] )?   # Non-greedy match for any comments in parentheses
            $                       # End of string
        """,
        re.VERBOSE,
    )
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

    def convert_object_to_int(
        self, data: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        for col in columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
        return data

    def parse_column_name(self, column: str, *, return_index: bool = False) -> str:
        if col_match := self.col_pattern.match(column):
            index = col_match.group('index')
            name = col_match.group('name')
            stage = col_match.group('stage')
            match stage:
                case '':
                    stage = ''
                case 'Ⅲ':
                    stage = '(III)'
                case _:
                    stage = f'({stage})'
            index = f'[{index}] ' if return_index else ''
            return f'{index}{name}{stage}'
        else:
            raise ValueError(
                f'Column name "{column}" does not match the expected pattern.'
            )

    def parse_column_values(
        self, data: pd.DataFrame, column: str
    ) -> tuple[pd.Series, list[tuple[int, str]]]:
        values = data[column]
        val_map = self.process_info(dict[str, str], column)
        if not self.config.final:
            for val in values.unique():
                if not pd.isna(val) and (val_match := self.val_pattern.match(val)):
                    value = val_match.group('value')
                    name = val_match.group('name')
                    val_map[value] = name
        values = values.replace(regex=self.val_pattern, value=r'\g<value>').astype(
            'Int64'
        )
        val_list = sorted((int(value), name) for value, name in val_map.items())
        return values, val_list

    def parse_column_codes(
        self, data: pd.DataFrame, column: str
    ) -> tuple[pd.Series, list[tuple[str, str]]]:
        codes = data[column]
        code_map = self.process_info(dict[str, str], column)
        if not self.config.final:
            for code in codes.unique():
                if not pd.isna(code):
                    for code_match in self.code_pattern.finditer(code):
                        code = code_match.group('code')
                        name = code_match.group('name')
                        code_map[code] = name
        codes.replace(regex=self.code_pattern, value=r'\g<code>', inplace=True)
        code_list = sorted(code_map.items())
        return codes, code_list

    def binarize_single_choice(
        self, data: pd.DataFrame, column: str, *, ordered: bool = True
    ) -> pd.DataFrame:
        values, val_list = self.parse_column_values(data, column)
        data = data.drop(columns=[column])
        stem = self.parse_column_name(column)
        if ordered:
            value, name = val_list[-1]
            if name == '其它':
                val_list = val_list[:-1]
                data[f'{stem} = {name}'] = (values == value).astype('Int8')
                values[
                    values == value
                ] = -1  # Temporarily set to -1 to avoid interference
            value, name = val_list[0]
            data[f'{stem} = {name}'] = (values == value).astype('Int8')
            for value, name in val_list[1:-1]:
                data[f'{stem} <= {name}'] = ((values <= value) & (values >= 0)).astype(
                    'Int8'
                )
                data[f'{stem} >= {name}'] = (values >= value).astype('Int8')
            value, name = val_list[-1]
            data[f'{stem} = {name}'] = (values == value).astype('Int8')
        else:
            for value, name in val_list:
                data[f'{stem} = {name}'] = (values == value).astype('Int8')
        return data

    def binarize_multiple_choice(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        codes, code_list = self.parse_column_codes(data, column)
        data = data.drop(columns=[column])
        stem = self.parse_column_name(column)
        for code, name in code_list:
            data[f'{stem} = {name}'] = codes.str.contains(code).astype('Int8')
        return data

    def binarize_onset_data(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        col = pd.to_numeric(data[column]).astype('Float64')
        data = data.drop(columns=[column])
        stem = self.parse_column_name(column)
        data[f'{stem} = 无'] = (col < 0).astype('Int8')
        if self.config.final:
            min_age = self.process_info(int, column, 'min')
            max_age = self.process_info(int, column, 'max')
        else:
            min_age, max_age = int(col[col >= 0].min()), int(col.max())
            self.process_info[column, 'min'] = min_age
            self.process_info[column, 'max'] = max_age
        for th in range(min_age // 5 * 5 + 4, max_age, 5):
            data[f'{stem} <= {th}岁'] = ((col <= th) & (col >= 0)).astype('Int8')
        for th in range(min_age // 5 * 5 + 5, max_age + 1, 5):
            data[f'{stem} >= {th}岁'] = (col >= th).astype('Int8')
        return data

    def process_continuous_data(
        self, data: pd.DataFrame, column: str, previous: str
    ) -> pd.DataFrame:
        name = self.parse_column_name(column)
        if name == '出现时间':
            prefix = self.parse_column_name(previous)
            name = f'{prefix}{name}'
        data.rename(columns={column: name}, inplace=True)
        return data

    def process_data(
        self, data: pd.DataFrame, threshold: float, unordered_columns: list[str]
    ) -> pd.DataFrame:
        # Process each column
        columns = data.columns.to_list()
        multiple_choice_keywords = ['可多选', '娱乐时间', '失眠形式']
        for col, prev in zip(columns, [columns[-1]] + columns[:-1], strict=False):
            if col == 'V14_发病时间':
                data = self.binarize_onset_data(data, col)
            elif any(kw in col for kw in multiple_choice_keywords):
                data = self.binarize_multiple_choice(data, col)
            elif col in unordered_columns:
                data = self.binarize_single_choice(data, col, ordered=False)
            elif is_string_dtype(data[col]):
                data = self.binarize_single_choice(data, col)
            else:
                data = self.process_continuous_data(data, col, prev)

        if not self.config.final:
            # Drop columns with less than `threshold` non-NaN values
            thresh = int(len(data) * threshold)
            data = data.dropna(axis=1, thresh=thresh)
            # Drop rows having any NaN values implicitly

        return data

    def get_life_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[
            :,
            'V22_Ⅰ婚前期，.当且仅当正常被试和看护者对此部分信息知晓才记录-I常居住地（可多选，居住时间超过一半的可视为常居住地）':'V124_工作总年限（年）',  # type: ignore
        ]

        # Convert object dtype columns to Int64
        object_columns = ['V62_II怀孕数', 'V63_II流产次数', 'V123_退休年龄（岁）']
        data = self.convert_object_to_int(data, object_columns)

        threshold = 0.88
        unordered_columns = [
            'V60_II婚姻状态',
            'V68_II婚姻幸福感评定人',
            'V81_Ⅲ婚姻状态',
            'V84_Ⅲ婚姻幸福感评定人',
            'V115_擅长文科还是理科',
            'V116_当前职业范围',
            'V120_既往职业',
        ]
        data = self.process_data(data, threshold, unordered_columns)

        return data

    def get_diet_and_medication_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[:, 'V125_是否吸烟':'V170_溶剂']  # type: ignore

        threshold = 0.89
        unordered_columns = data.loc[:, 'V136_避孕药':'V170_溶剂'].columns.to_list()  # type: ignore
        data = self.process_data(data, threshold, unordered_columns)

        return data

    def get_family_history_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[
            :, 'V172_痴呆':'V217_家族中是否有脾气性格难相处，行为异常的成员'  # type: ignore
        ]

        # Fit the val_pattern
        data.replace('4(5>3人)', '4(5=超过3人)', inplace=True)

        # Rename conflicted columns
        data.rename(
            columns={
                'V190_抑郁（须有明确诊断）': 'V190_抑郁家族史',
                'V194_高血压': 'V194_高血压家族史',
                'V198_糖尿病': 'V198_糖尿病家族史',
            },
            inplace=True,
        )

        threshold = 0.86
        unordered_columns = [
            'V173_痴呆家族史',
            'V174_痴呆家族史1',
            'V176_神经科其它疾病家族史1',
        ]
        data = self.process_data(data, threshold, unordered_columns)

        return data

    def get_medical_history_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[:, 'V218_物质依赖史':'V311_腰臀比(腰围/臀围)']  # type: ignore

        # Reorder some values
        data.replace(
            {'V240_体重变化': {'0(1=无)': '1(2=无)', '1(2=减少)': '0(1=减少)'}},
            inplace=True,
        )

        threshold = 0.9
        unordered_columns = [
            'V238_甲状腺功能',
            'V279_精神疾病',
            'V280_是否有视力障碍',
            'V297_头痛',
        ]
        data = self.process_data(data, threshold, unordered_columns)

        return data

    def get_symptom_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = pd.concat(
            [
                data.loc[:, 'V14_发病时间'],
                data.loc[:, 'V312_首发症状':'V393_病程经过（可多选）'],  # type: ignore
            ],
            axis=1,
        )

        # Fill data for normal people
        fill_values = {col: '0(a=无)' for col in data.columns}
        fill_values['V14_发病时间'] = '-1'
        fill_values['V312_首发症状'] = '4(n=无)'
        data.replace('无', fill_values, inplace=True)

        # Reorder some values
        replace_pattern = {
            '0(a=无)': '1(b=无)',
            '1(b=下降)': '0(a=下降)',
            '1(b=减少)': '0(a=减少)',
        }
        replace_columns = ['V373_食欲改变', 'V377_夜间睡眠改变', 'V378_白天睡眠改变']
        data.replace({col: replace_pattern for col in replace_columns}, inplace=True)

        threshold = 0.0
        unordered_columns = ['V312_首发症状']
        data = self.process_data(data, threshold, unordered_columns)

        return data

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()

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
