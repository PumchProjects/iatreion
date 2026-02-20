from pathlib import Path

from iatreion.exceptions import IatreionException
from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig


def get_exp_root(name: str, train: TrainConfig) -> Path:
    groups_root = (
        train.log_root
        / name
        / train.group_name_str
        / 'rrl'
        / ('final' if train.final else train.ref_name_str)
    )
    if not groups_root.is_dir():
        raise IatreionException(
            'No experiment root found for $dataset and groups "$groups".',
            dataset=name,
            groups=train.group_name_str,
        )
    return groups_root


def get_rrl_file(exp_root: Path, train: TrainConfig) -> Path:
    return exp_root / (
        f'{train.cur_name}.tsv'
        if train.final
        else f'{train.cur_name}_{train.ith_kfold}.tsv'
    )


def register_log_dir(
    dataset: DatasetConfig,
    train: TrainConfig,
    model_name: str,
    folder_name: str | None = None,
    file_name: str = 'train.log',
) -> None:
    train.log_dir = (
        train.log_root
        / dataset.name_str
        / train.group_name_str
        / model_name
        / ('final' if train.final else train.ref_name_str)
    )
    if folder_name is not None and not train.final:
        train.log_dir /= folder_name
    add_file_handler(train.log_dir / file_name)
