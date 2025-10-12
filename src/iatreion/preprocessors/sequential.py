from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig
from iatreion.utils import logger

from .base import NamedPreprocessor, Preprocessor


class SequentialPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, children: list[NamedPreprocessor]
    ) -> None:
        super().__init__(config)
        self.children = children

    @override
    def get_data(self) -> pd.DataFrame:
        name, child = self.children[0]
        data = self.get_child_data(name, child, copy_indices=True)
        logger.info(f'Got `{name}` data!')
        for name, child in self.children[1:]:
            child_data = self.get_child_data(name, child)
            logger.info(f'Merging `{name}` with previous data...')
            if self.config.final:
                data = pd.concat([data, child_data], axis=1)
            else:
                data = data.merge(child_data, left_index=True, right_index=True)
        return data
