from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy.typing import NDArray

from iatreion.utils import SubprocessWorker

_WORKER_SCRIPT = Path(__file__).with_name('limix_worker.py')


@dataclass(frozen=True, slots=True)
class LimiXWorkerConfig:
    python_path: Path
    repo_path: Path
    model_path: Path
    inference_config_path: Path
    device: str = 'cuda'
    mask_prediction: bool = False


class LimiXWorkerClient:
    def __init__(
        self, config: LimiXWorkerConfig, *, name: str = 'LimiX worker'
    ) -> None:
        argv = [
            str(config.python_path),
            str(_WORKER_SCRIPT),
            '--repo-path',
            str(config.repo_path),
            '--model-path',
            str(config.model_path),
            '--inference-config',
            str(config.inference_config_path),
            '--device',
            config.device,
        ]
        if config.mask_prediction:
            argv.append('--mask-prediction')
        self._worker = SubprocessWorker(
            argv,
            cwd=config.repo_path,
            name=name,
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
        *,
        task_type: str,
    ) -> Any:
        if self._worker.ensure_started():
            self._fit_loaded = False
        if not self._fit_loaded:
            self._request_ok('fit', payload=(X_train, y_train))
            self._fit_loaded = True

        payload: dict[str, Any] = {'X_test': X_test, 'task_type': task_type}
        return self._request_ok('predict', payload=payload)

    def close(self) -> None:
        self._worker.close()
        self._fit_loaded = False

    def _request_ok(self, command: str, *, payload: Any = None) -> Any:
        response = self._worker.request({'command': command, 'payload': payload})
        status = response.get('status') if isinstance(response, dict) else None
        if status != 'ok':
            raise RuntimeError(self._worker.format_response_error(response))
        return response.get('result')
