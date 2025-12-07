import subprocess
from typing import Literal, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from iatreion.configs import LimiXConfig
from iatreion.utils import chdir

from .base import Model, ModelReturn


class LimiXModel(Model):
    def __init__(self, config: LimiXConfig) -> None:
        super().__init__()
        self.config = config

    def save_dataset(
        self, X: NDArray, y: NDArray, mode: Literal['train', 'test']
    ) -> None:
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        columns = [f'V{i}' for i in range(X.shape[1])] + ['target']
        df = pd.DataFrame(data, columns=columns)
        match mode:
            case 'train':
                df.to_csv(self.config.train_file, index=False)
            case 'test':
                df.to_csv(self.config.test_file, index=False)

    def run_inference(self) -> None:
        cmd = [
            str(self.config.python_path),
            str(self.config.script_path),
            '--data_dir',
            str(self.config.data_dir),
            '--save_name',
            self.config.save_name,
            '--inference_config_path',
            str(self.config.inference_config_path),
            '--model_path',
            str(self.config.model_path),
        ]
        subprocess.run(cmd, check=True)

    def get_score(self) -> NDArray:
        result_df = pd.read_csv(self.config.result_file)
        y_score = result_df.iloc[:, 1:].to_numpy()
        return y_score

    @override
    def fit(self, X: NDArray, y: NDArray) -> None:
        self.save_dataset(X, y, mode='train')

    @override
    def predict(self, X: NDArray, y: NDArray) -> ModelReturn:
        self.save_dataset(X, y, mode='test')
        with chdir(self.config.repo_path):
            self.run_inference()
        y_score = self.get_score()
        return y_score, {}
