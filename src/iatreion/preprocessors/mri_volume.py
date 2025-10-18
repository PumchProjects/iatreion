from typing import Literal, override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class VolumePreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        columns = [col for col in data.columns if col.endswith('_Z')]
        data = data[columns].dropna(axis=1, how='all')
        return data


def get_feature(name: DataName) -> Literal['v', 'pct']:
    feature = name.rsplit('-', maxsplit=1)[-1]
    assert feature in ('v', 'pct')
    return feature


class VolumeAveragePreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, name: DataName, *, use_z: bool = False
    ) -> None:
        super().__init__(config, name)
        feature = get_feature(name)
        self.feature = f'{feature}_Z' if use_z else feature

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = data.loc[:, ~data.columns.str.startswith('Brainstem')]
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
        a_columns = []
        for col in lr_columns:
            a_columns.append(f'{col}_A_{self.feature}')
            data[f'{col}_A_{self.feature}'] = (
                data[f'{col}_L_{self.feature}'] + data[f'{col}_R_{self.feature}']
            ) * 0.5
        data = data[c_columns + a_columns + ai_columns]
        return data


class VolumeAverageNewPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, name: DataName, *, new: bool = False
    ) -> None:
        super().__init__(config, name)
        feature = get_feature(name)
        if new:
            self.feature = '(cm³)' if feature == 'v' else '(%)'
            self.left_suffix = f'(L){self.feature}'
            self.right_suffix = f'(R){self.feature}'
            self.mri_time_col = '检查日期/Study date'
        else:
            self.feature = f'_{feature}'
            self.left_suffix = f'_L{self.feature}'
            self.right_suffix = f'_R{self.feature}'
            self.mri_time_col = 'MRI_time'
        self.new = new
        self.placeholder = '<>'

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
        if not self.new:
            data = data.loc[:, :'CC_Posterior_pct']  # type: ignore
        data, real_ages = self.calc_ages(data, self.mri_time_col, force_final=self.new)
        data['age_group'] = real_ages.apply(self.match_group).astype('string')
        return data

    def rename_vmri(self, vmri: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        change = pd.read_excel(self.config.vmri_change_path, index_col='原表头名称')
        change_dict = change['新表头名称'].dropna().to_dict()
        for key in vmri:
            vmri[key].rename(columns=change_dict, inplace=True)
        return vmri

    def get_mean_std(self, data: pd.DataFrame) -> pd.DataFrame:
        vmri = pd.read_excel(
            self.config.vmri_data_path,
            sheet_name=['mean', 'sd'],
            dtype_backend='numpy_nullable',
        )
        if self.new:
            vmri = self.rename_vmri(vmri)
        data.reset_index(inplace=True)
        data = data.merge(
            vmri['mean'], how='left', on='age_group', suffixes=(None, '_mean')
        )
        data = data.merge(
            vmri['sd'], how='left', on='age_group', suffixes=(None, '_std')
        )
        data = data.loc[:, ~data.columns.str.startswith('Brainstem')]
        data.set_index(self.index_name, inplace=True)
        return data

    def extract_stem(self, left_col: str) -> str:
        if self.new:
            left_col = left_col.replace('_左', self.placeholder)
        return left_col.removesuffix(self.left_suffix)

    def get_name(self, col: str) -> str:
        return col.split('/', maxsplit=1)[0].removesuffix(self.placeholder)

    def recover(self, stem: str, side: Literal['left', 'right']) -> str:
        if self.new:
            stem = stem.replace(self.placeholder, '_左' if side == 'left' else '_右')
        return f'{stem}{self.left_suffix if side == "left" else self.right_suffix}'

    def get_columns(self, data: pd.DataFrame) -> tuple[list[str], ...]:
        columns = [col for col in data.columns if col.endswith(self.feature)]
        c_columns = [
            col
            for col in columns
            if not col.endswith(self.left_suffix)
            and not col.endswith(self.right_suffix)
        ]
        lr_columns = [
            self.extract_stem(col) for col in columns if col.endswith(self.left_suffix)
        ]
        return c_columns, lr_columns

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
            z_col = self.get_name(col) if self.new else f'{col}_Z'
            data[z_col] = self.calc_z_score(data, col)
            z_columns.append(z_col)
        return z_columns

    def calc_average_z_scores(
        self, data: pd.DataFrame, columns: list[str]
    ) -> list[str]:
        z_columns = []
        for col in columns:
            z_col = (
                f'{self.get_name(col)}_平均' if self.new else f'{col}_A{self.feature}_Z'
            )
            left = self.calc_z_score(data, self.recover(col, 'left'))
            right = self.calc_z_score(data, self.recover(col, 'right'))
            data[z_col] = (left + right) * 0.5
            z_columns.append(z_col)
        return z_columns

    def calc_asymmetry_indices(
        self, data: pd.DataFrame, columns: list[str]
    ) -> list[str]:
        if not self.new:
            return [f'{col}_Asymmetry_index' for col in columns]
        ai_columns = []
        for col in columns:
            ai_col = f'{self.get_name(col)}_不对称指数'
            left = data[self.recover(col, 'left')]
            right = data[self.recover(col, 'right')]
            data[ai_col] = (left - right).abs() / ((left + right) * 0.5)
            ai_columns.append(ai_col)
        return ai_columns

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = self.calc_age_groups(data)
        data = self.get_mean_std(data)
        c_columns, lr_columns = self.get_columns(data)
        cz_columns = self.calc_central_z_scores(data, c_columns)
        az_columns = self.calc_average_z_scores(data, lr_columns)
        ai_columns = self.calc_asymmetry_indices(data, lr_columns)
        data = data[cz_columns + az_columns + ai_columns]
        return data
