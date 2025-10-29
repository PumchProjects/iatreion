from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter

from .dataset import DatasetConfig
from .train import TrainConfig
from .utils import get_best_exp_root, get_rrl_file, register_log_dir


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig:
    dataset: DatasetConfig

    train: TrainConfig

    def __post_init__(self) -> None:
        self.dataset.simple = False
        if not self.train.final:
            file_name = 'val.log' if self.train.val_size is not None else 'test.log'
            register_log_dir(
                self.dataset, self.train, 'rrl-discrete', file_name=file_name
            )

    def get_best_exp_roots(self) -> list[Path]:
        exp_roots: list[Path] = []
        for name in self.dataset.names:
            exp_roots.append(get_best_exp_root(name, self.train))
        return exp_roots

    def get_rrl_file(self, exp_root: Path) -> Path:
        return get_rrl_file(exp_root, self.train)
