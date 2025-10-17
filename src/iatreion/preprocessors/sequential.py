from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig
from iatreion.utils import logger

from .base import Preprocessor


class SequentialPreprocessor(Preprocessor):
    def __init__(
        self, config: PreprocessorConfig, name: DataName, children: list[Preprocessor]
    ) -> None:
        super().__init__(config, name)
        self.children = children

    @override
    def get_data(self) -> pd.DataFrame:
        child = self.children[0]
        data = child.get_data_outer(dedup=True)
        logger.info(f'Got "{child.name}" data!')
        for child in self.children[1:]:
            child_data = child.get_data_outer(dedup=True)
            logger.info(f'Merging "{child.name}" with previous data...')
            if self.config.final:
                data = pd.concat([data, child_data], axis=1)
            else:
                data = data.merge(child_data, left_index=True, right_index=True)
        return data
