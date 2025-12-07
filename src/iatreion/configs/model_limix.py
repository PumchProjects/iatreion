from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter
from cyclopts.types import Directory, ExistingDirectory, ExistingFile

from .dataset import DatasetConfig
from .train import TrainConfig
from .utils import register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class LimiXConfig:
    dataset: DatasetConfig

    train: TrainConfig

    data_root: Directory
    'Directory to store LimiX data files.'

    python_path: ExistingFile
    'Path to the Python interpreter that supports LimiX.'

    repo_path: ExistingDirectory
    'Path to the LimiX repository.'

    model_path: ExistingFile
    'Path to the pre-trained LimiX model file.'

    data_prefix: str = 'tmp'
    'Name prefix for LimiX data files.'

    @property
    def data_dir(self) -> Path:
        return self.data_root / str(self.train.device_id)

    @property
    def data_subdir(self) -> Path:
        return self.data_dir / self.data_prefix

    @property
    def train_file(self) -> Path:
        return self.data_subdir / f'{self.data_prefix}_train.csv'

    @property
    def test_file(self) -> Path:
        return self.data_subdir / f'{self.data_prefix}_test.csv'

    @property
    def script_path(self) -> Path:
        return self.repo_path / 'inference_classifier.py'

    @property
    def inference_config_path(self) -> Path:
        return self.repo_path / 'config' / 'cls_default_noretrieval.json'

    @property
    def save_name(self) -> str:
        return str(self.train.device_id)

    @property
    def result_file(self) -> Path:
        return (
            self.repo_path
            / 'result'
            / self.save_name
            / f'{self.data_prefix}_pred_LimiX.csv'
        )

    def __post_init__(self) -> None:
        self.dataset.simple = True
        register_log_dir(self.dataset, self.train, 'limix')
        self.data_subdir.mkdir(parents=True, exist_ok=True)
