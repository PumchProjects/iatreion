from typing import override

from numpy.typing import NDArray

from iatreion.configs import LimiXConfig
from iatreion.train_utils import LimiXWorkerClient, LimiXWorkerConfig

from .base import Model


class LimiXModel(Model):
    def __init__(self, config: LimiXConfig) -> None:
        super().__init__()
        self.config: LimiXConfig = config
        self.num_class = config.train.num_class
        self._train_data: tuple[NDArray, NDArray] | None = None
        self._worker = LimiXWorkerClient(
            LimiXWorkerConfig(
                python_path=config.train.limix_python_path,
                repo_path=config.train.limix_repo_path,
                model_path=config.train.limix_model_path,
                inference_config_path=config.inference_config_path,
                device=config.train.limix_device,
            )
        )

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self._train_data = (X, y)
        self._worker.mark_dirty()

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        if self._train_data is None:
            raise RuntimeError('LimiXModel must be fitted before prediction.')
        X_train, y_train = self._train_data
        return self._worker.predict(X, X_train, y_train, task_type='Classification')

    @override
    def close(self) -> None:
        self._worker.close()
