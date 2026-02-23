from dataclasses import dataclass
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

    def get_exp_root(self, name: str) -> Path:
        groups_root = (
            self.train.log_root
            / name
            / self.train.group_name_str
            / 'rrl'
            / ('final' if self.train.final else self.train.ref_name_str)
        )
        if not groups_root.is_dir():
            raise IatreionException(
                'No experiment root found for $dataset and groups "$groups".',
                dataset=name,
                groups=self.train.group_name_str,
            )
        return groups_root

    def register_log_dir(
        self,
        model_name: str,
        *,
        folder_name: str | None = None,
        file_name: str = 'train.log',
    ) -> None:
        self.train.log_dir = (
            self.train.log_root
            / self.dataset.name_str
            / self.train.group_name_str
            / model_name
            / ('final' if self.train.final else self.train.ref_name_str)
        )
        if folder_name is not None and not self.train.final:
            self.train.log_dir /= folder_name
        add_file_handler(self.train.log_dir / file_name)
