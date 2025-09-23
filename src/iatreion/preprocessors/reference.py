from typing import override

import pandas as pd

from iatreion.configs import PreprocessorConfig
from iatreion.utils import logger

from .base import NamedPreprocessor, Preprocessor


class ReferencePreprocessor(Preprocessor):
    def __init__(
        self,
        config: PreprocessorConfig,
        child: NamedPreprocessor,
        ref_child: NamedPreprocessor,
    ) -> None:
        super().__init__(config)
        self.child = child
        self.ref_child = ref_child

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.get_child_data(*self.child)
        logger.info(f'Got `{self.child[0]}` data!')
        ref_data = self.get_child_data(*self.ref_child)
        logger.info(f'Referencing to `{self.ref_child[0]}` data...')
        data = data[data.index.isin(ref_data.index)]
        return data
