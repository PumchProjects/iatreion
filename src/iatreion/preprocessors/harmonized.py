from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class CategoricalPreprocessor(Preprocessor):
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
        self, config: PreprocessorConfig, name: DataName, *, prefix: str
    ) -> None:
        super().__init__(config, name)
        self.prefix = prefix

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        selected = [col for col in data.columns if col.startswith(self.prefix)]
        return data[selected]


class HarmonizedDemoPreprocessor(CategoricalPreprocessor):
    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        data = self.make_categorical(data, ['sex', 'handedness'])
        selected = ['sex', 'age_at_visit', 'edu_year', 'handedness']
        return data[selected]


class HarmonizedApoePreprocessor(Preprocessor):
    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        return data[['apoe4']]


class HarmonizedMriPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, name: DataName, *, roi: bool
    ) -> None:
        super().__init__(config, name)
        self.roi = roi

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        selected = [
            col
            for col in data.columns
            if col.endswith('_v_w') or col.endswith('_asymmetry_index')
        ]
        if self.roi:
            selected = [col for col in selected if col.startswith('ROI_')]
        else:
            selected = [col for col in selected if not col.startswith('ROI_')]
        return data[selected]


class HarmonizedHistoryPreprocessor(CategoricalPreprocessor):
    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()
        selected = ['baseline_insomnia', 'hypertension_history']
        selected += [col for col in data.columns if col.startswith('mh_')]
        data = self.make_categorical(data, selected)
        return data[selected]
