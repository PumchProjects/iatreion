from typing import Literal, override

import pandas as pd

from iatreion.configs import PreprocessorConfig

from .base import Preprocessor


class VolumePreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig) -> None:
        super().__init__(config)

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        columns = [col for col in data.columns if col.endswith('_Z')]
        data = data[columns].dropna(axis=1, how='all').dropna()
        return data


class VolumeAveragePreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, feature: Literal['v', 'pct']
    ) -> None:
        super().__init__(config)
        self.feature = feature

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        ai_columns = [col for col in data.columns if col.endswith('_Asymmetry_index')]
        columns = [col for col in data.columns if col.endswith(f'_{self.feature}_Z')]
        c_columns = [
            col
            for col in columns
            if not col.endswith(f'_L_{self.feature}_Z')
            and not col.endswith(f'_R_{self.feature}_Z')
        ]
        lr_columns = [
            col.removesuffix(f'_L_{self.feature}_Z')
            for col in columns
            if col.endswith(f'_L_{self.feature}_Z')
        ]
        a_columns = []
        for col in lr_columns:
            a_columns.append(f'{col}_A_{self.feature}_Z')
            data[f'{col}_A_{self.feature}_Z'] = (
                data[f'{col}_L_{self.feature}_Z'] + data[f'{col}_R_{self.feature}_Z']
            ) * 0.5
        data = (
            data[c_columns + a_columns + ai_columns].dropna(axis=1, how='all').dropna()
        )
        return data


class VolumeAverageNewPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, feature: Literal['v', 'pct']
    ) -> None:
        super().__init__(config)
        self.feature = feature

    @staticmethod
    def match_group(age: int | float) -> str | None:
        match age:
            case age if 50 <= age <= 54:
                return '50-54'
            case age if 55 <= age <= 59:
                return '55-59'
            case age if 60 <= age <= 64:
                return '60-64'
            case age if 65 <= age <= 69:
                return '65-69'
            case age if 70 <= age <= 74:
                return '70-74'
            case age if 75 <= age <= 79:
                return '75-79'
            case age if 80 <= age <= 84:
                return '80-84'
            case age if age >= 85:
                return '>=85'
            case _:
                return None

    def calc_age_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[:, 'MRI_time':'CC_Posterior_pct']  # type: ignore
        data = data.dropna(subset=['MRI_time'])
        birth_dates = self.get_birth_dates()
        data = data.merge(birth_dates, left_index=True, right_index=True, copy=False)
        MRI_time = pd.to_datetime(data['MRI_time'], utc=True)
        real_ages = (MRI_time - data['birth_date']).dt.days // 365.2422
        data['age_group'] = real_ages.apply(self.match_group)
        data = data.dropna(subset=['age_group'])
        return data

    def get_mean_std(self, data: pd.DataFrame) -> pd.DataFrame:
        vmri = pd.read_excel(self.config.vmri_data_path, sheet_name=['mean', 'sd'])
        data = data.reset_index()
        data = data.merge(
            vmri['mean'], on='age_group', suffixes=(None, '_mean'), copy=False
        )
        data = data.merge(
            vmri['sd'], on='age_group', suffixes=(None, '_std'), copy=False
        )
        data = data.loc[:, ~data.columns.str.startswith('Brainstem')]
        return data.set_index('serial_num')

    def get_columns(self, data: pd.DataFrame) -> tuple[list[str], ...]:
        ai_columns = [col for col in data.columns if col.endswith('_Asymmetry_index')]
        columns = [col for col in data.columns if col.endswith(f'_{self.feature}')]
        c_columns = [
            col
            for col in columns
            if not col.endswith(f'_L_{self.feature}')
            and not col.endswith(f'_R_{self.feature}')
        ]
        lr_columns = [
            col.removesuffix(f'_L_{self.feature}')
            for col in columns
            if col.endswith(f'_L_{self.feature}')
        ]
        return c_columns, lr_columns, ai_columns

    @staticmethod
    def calc_z_score(data: pd.DataFrame, feature: str) -> pd.Series:
        mean = data[f'{feature}_mean']
        std = data[f'{feature}_std']
        return (data[feature] - mean) / std

    def calc_central_z_scores(
        self, data: pd.DataFrame, columns: list[str]
    ) -> list[str]:
        z_columns = []
        for col in columns:
            z_col = f'{col}_Z'
            data[z_col] = self.calc_z_score(data, col)
            z_columns.append(z_col)
        return z_columns

    def calc_average_z_scores(
        self, data: pd.DataFrame, columns: list[str]
    ) -> list[str]:
        z_columns = []
        for col in columns:
            z_col = f'{col}_A_{self.feature}_Z'
            data[z_col] = (
                self.calc_z_score(data, f'{col}_L_{self.feature}')
                + self.calc_z_score(data, f'{col}_R_{self.feature}')
            ) * 0.5
            z_columns.append(z_col)
        return z_columns

    @override
    def get_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.config.data_path, index_col='serial_num')
        data = self.calc_age_groups(data)
        data = self.get_mean_std(data)
        c_columns, lr_columns, ai_columns = self.get_columns(data)
        cz_columns = self.calc_central_z_scores(data, c_columns)
        az_columns = self.calc_average_z_scores(data, lr_columns)
        data = data[cz_columns + az_columns + ai_columns].dropna()
        return data
