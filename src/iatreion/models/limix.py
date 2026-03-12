import base64
import pickle
import subprocess
from typing import override

from numpy.typing import NDArray

from iatreion.configs import LimiXConfig
from iatreion.utils import chdir

from .base import Model

wrapped_code = """
import sys

sys.path.append()
"""


class LimiXModel(Model):
    def __init__(self, config: LimiXConfig) -> None:
        super().__init__()
        self.config: LimiXConfig = config
        self.num_class = config.train.num_class
        self.input_data = {}

    def _register(self, data: NDArray, name: str) -> None:
        self.input_data[name] = data

    @staticmethod
    def _serialize(data: NDArray) -> str:
        return base64.b64encode(pickle.dumps(data)).decode('utf-8')

    @staticmethod
    def _deserialize(data_b64: str) -> NDArray:
        return pickle.loads(base64.b64decode(data_b64))

    def _run_inference(self) -> NDArray:
        input_b64 = self._serialize(self.input_data)
        wrapped_code = f"""
import base64, pickle, torch
from inference.predictor import LimiXPredictor
input_data = pickle.loads(base64.b64decode('{input_b64}'))
for k, v in input_data.items(): globals()[k] = v
y_score = LimiXPredictor(
    device=torch.device('cuda'),
    model_path='{self.config.model_path}',
    inference_config='{self.config.inference_config_path}',
).predict(X_train, y_train, X_test)
print(base64.b64encode(pickle.dumps(y_score)).decode('utf-8'))
"""
        process = subprocess.run(
            str(self.config.python_path),
            capture_output=True,
            check=True,
            encoding='utf-8',
            input=wrapped_code,
            text=True,
        )
        return self._deserialize(process.stdout.strip())

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self._register(X, 'X_train')
        self._register(y, 'y_train')

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        self._register(X, 'X_test')
        with chdir(self.config.repo_path):
            y_score = self._run_inference()
        return y_score
