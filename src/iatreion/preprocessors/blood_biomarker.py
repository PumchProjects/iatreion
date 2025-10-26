from typing import override

import pandas as pd

from iatreion.configs import DataName, PreprocessorConfig

from .base import Preprocessor


class BiomarkerPreprocessor(Preprocessor):
    def __init__(self, config: PreprocessorConfig, name: DataName) -> None:
        super().__init__(config, name)

    @override
    def get_data(self) -> pd.DataFrame:
        data = self.read_data()

        selected = ['Aβ42', 'ptau217', 'GFAP', 'NFL']
        for col in selected:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        divisor = selected[0]
        for dividend in selected[1:]:
            ratio_name = f'{dividend}/{divisor}'
            data[ratio_name] = data[dividend] / data[divisor]
            selected.append(ratio_name)

        return data[selected]
