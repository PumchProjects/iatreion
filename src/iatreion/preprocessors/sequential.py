from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class SequentialPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, children: list[tuple[DataName, Preprocessor]]
    ) -> None:
        super().__init__(config)
        self.children = children
        self.original_name = config.dataset.name

    def get_child_data(self, name: DataName, child: Preprocessor) -> pd.DataFrame:
        self.config.dataset.name = name
        data = child.get_data()
        self.config.dataset.name = self.original_name
        return data

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.get_child_data(*self.children[0])
        for name, child in self.children[1:]:
            child_data = self.get_child_data(name, child)
            data = data.merge(child_data, left_index=True, right_index=True, copy=False)
        return data
