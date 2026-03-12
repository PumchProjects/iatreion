from pathlib import Path
from typing import Any, override

from numpy.typing import NDArray

from iatreion.configs import LimiXConfig
from iatreion.utils.worker import SubprocessWorker

from .base import Model

_WORKER_SCRIPT = Path(__file__).with_name('limix_worker.py')


class _LimiXWorkerClient:
    def __init__(self, config: LimiXConfig) -> None:
        self._worker = SubprocessWorker(
            [
                str(config.python_path),
                str(_WORKER_SCRIPT),
                '--repo-path',
                str(config.repo_path),
                '--model-path',
                str(config.model_path),
                '--inference-config',
                str(config.inference_config_path),
            ],
            cwd=config.repo_path,
            name='LimiX worker',
            ready_status='ready',
            shutdown_request={'command': 'shutdown'},
        )
        self._fit_loaded = False

    def mark_dirty(self) -> None:
        self._fit_loaded = False

    def predict(
        self,
        X_test: NDArray,
        X_train: NDArray,
        y_train: NDArray,
    ) -> NDArray:
        if self._worker.ensure_started():
            self._fit_loaded = False
        if not self._fit_loaded:
            self._request_ok('fit', payload=(X_train, y_train))
            self._fit_loaded = True
        return self._request_ok('predict', payload=X_test)

    def close(self) -> None:
        self._worker.close()
        self._fit_loaded = False

    def _request_ok(self, command: str, *, payload: Any = None) -> Any:
        response = self._worker.request({'command': command, 'payload': payload})
        status = response.get('status') if isinstance(response, dict) else None
        if status != 'ok':
            raise RuntimeError(self._worker.format_response_error(response))
        return response.get('result')


class LimiXModel(Model):
    def __init__(self, config: LimiXConfig) -> None:
        super().__init__()
        self.config: LimiXConfig = config
        self.num_class = config.train.num_class
        self._train_data: tuple[NDArray, NDArray] | None = None
        self._worker = _LimiXWorkerClient(config)

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self._train_data = (X, y)
        self._worker.mark_dirty()

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        if self._train_data is None:
            raise RuntimeError('LimiXModel must be fitted before prediction.')
        X_train, y_train = self._train_data
        return self._worker.predict(X, X_train, y_train)

    @override
    def close(self) -> None:
        self._worker.close()
