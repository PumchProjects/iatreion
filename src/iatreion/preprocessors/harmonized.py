from typing import override

import numpy as np
import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class CategoricalPreprocessor(Preprocessor):
    def cut_ages(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for col in columns:
            data[col] = pd.cut(data[col], bins=list(range(0, 101, 5)))
        return data

    def make_categorical(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for col in columns:
            if self.config._final:
                categories = self.process_info(list[str], col, 'categories')
                data[col] = pd.Categorical(data[col], categories=categories)
            else:
                data[col] = data[col].astype('category')
                self.process_info[col, 'categories'] = data[col].cat.categories.tolist()
        return data


class PrefixPreprocessor(Preprocessor):
    def __init__(
        self,
        config: PreprocessorConfig,
        name: DataName,
        *,
        prefix: str,
        exceptions: list[str] | None = None,
    ) -> None:
        super().__init__(config, name)
        self.prefix = prefix
        self.exceptions = exceptions

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        if self.exceptions is not None:
            data = self.drop_columns(data, self.exceptions)
        selected = [col for col in data.columns if col.startswith(self.prefix)]
        return data[selected]


class HarmonizedDemoPreprocessor(CategoricalPreprocessor):
    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = self.cut_ages(data, ['age_at_visit', 'onset_age'])
        data = self.make_categorical(data, ['sex', 'handedness'])
        selected = ['sex', 'age_at_visit', 'onset_age', 'edu_year', 'handedness']
        return data[selected]


class HarmonizedApoePreprocessor(Preprocessor):
    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        return data[['apoe4']]


class HarmonizedMmsePreprocessor(PrefixPreprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(
            config,
            name,
            prefix='MMSE_',
            exceptions=['MMSE_slope_change_per_year', 'MMSE_progression_group'],
        )


class HarmonizedMocaPreprocessor(PrefixPreprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name, prefix='MOCA_')


class HarmonizedMriPreprocessor(Preprocessor):
    def get_columns(self, data: pd.DataFrame) -> tuple[list[str], ...]:
        columns = [col for col in data.columns if col.endswith('_v_w')]
        c_columns = [
            col
            for col in columns
            if not col.endswith('_L_v_w') and not col.endswith('_R_v_w')
        ]
        lr_columns = [
            col.removesuffix('_L_v_w') for col in columns if col.endswith('_L_v_w')
        ]
        return c_columns, lr_columns

    def calc_average_scores(
        self, data: pd.DataFrame, lr_columns: list[str]
    ) -> list[str]:
        a_columns = []
        for col in lr_columns:
            a_col = f'{col}_A_v_w'
            data[a_col] = (data[f'{col}_L_v_w'] + data[f'{col}_R_v_w']) * 0.5
            a_columns.append(a_col)
        return a_columns

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        c_columns, lr_columns = self.get_columns(data)
        a_columns = self.calc_average_scores(data, lr_columns)
        ai_columns = [col for col in data.columns if col.endswith('_asymmetry_index')]
        selected = [
            col
            for col in c_columns + a_columns + ai_columns
            if not col.startswith('ROI_')
        ]
        return data[selected]


class HarmonizedMriRoiPreprocessor(PrefixPreprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name, prefix='ROI_')


class HarmonizedPlasmaPreprocessor(PrefixPreprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name, prefix='Plasma_')


class HarmonizedLabdataPreprocessor(PrefixPreprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name, prefix='LABDATA_', exceptions=['LABDATA_DATE'])


class HarmonizedHistoryPreprocessor(CategoricalPreprocessor):
    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data.replace('Unknown', np.nan, inplace=True)
        selected = ['baseline_insomnia', 'hypertension_history']
        selected += [col for col in data.columns if col.startswith('mh_')]
        data = self.make_categorical(data, selected)
        return data[selected]
