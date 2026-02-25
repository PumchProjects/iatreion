from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from cyclopts import Parameter

from iatreion.exceptions import IatreionException
from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ModelConfig:
    dataset: DatasetConfig
    train: TrainConfig

    def get_exp_root(self, model_name: str) -> Path:
        return (
            self.train.log_root
            / ('final' if self.train.final else self.dataset.name_str)
            / self.train.group_name_str
            / model_name
            / ('' if self.train.final else self.train.ref_name_str)
        )

    @cached_property
    def rrl_root(self) -> Path:
        exp_root = self.get_exp_root('rrl')
        if not exp_root.is_dir():
            raise IatreionException(
                'No experiment root found for $dataset and groups "$groups".',
                dataset='final' if self.train.final else self.dataset.name_str,
                groups=self.train.group_name_str,
            )
        return exp_root

    def register_log_dir(
        self,
        model_name: str,
        *,
        folder_name: str | None = None,
        file_name: str = 'train.log',
    ) -> None:
        self.train._log_dir = self.get_exp_root(model_name)
        if folder_name is not None and not self.train.final:
            self.train._log_dir /= folder_name
        add_file_handler(self.train._log_dir / file_name)
