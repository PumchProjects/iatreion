import re
from pathlib import Path
from typing import Literal, overload

from iatreion.exceptions import IatreionException
from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig

avg_f1_pattern = re.compile(r'AVG F1\s+(?P<value>.+?)%')


def get_avg_f1(exp_root: Path) -> float:
    file = exp_root / 'train.log'
    if not file.exists():
        return 0.0
    data = file.read_text(encoding='utf-8')
    match = avg_f1_pattern.search(data)
    if match:
        return float(match.group('value')) / 100.0
    return 0.0


@overload
def try_get_best_exp_root(
    groups_root: Path, final: Literal[False]
) -> tuple[Path, float] | None: ...


@overload
def try_get_best_exp_root(
    groups_root: Path, final: Literal[True]
) -> tuple[Path, None] | None: ...


def try_get_best_exp_root(
    groups_root: Path, final: bool
) -> None | tuple[Path, None] | tuple[Path, float]:
    if not groups_root.is_dir():
        return None
    if final:
        return groups_root, None
    best_f1 = 0.0
    best_exp_root: Path | None = None
    for exp_root in groups_root.iterdir():
        f1 = get_avg_f1(exp_root)
        if f1 > best_f1:
            best_f1 = f1
            best_exp_root = exp_root
    if best_exp_root is not None:
        return best_exp_root, best_f1
    return None


@overload
def get_best_exp_root(
    name: str, train: TrainConfig, final: Literal[False]
) -> tuple[Path, float]: ...


@overload
def get_best_exp_root(
    name: str, train: TrainConfig, final: Literal[True]
) -> tuple[Path, None]: ...


def get_best_exp_root(
    name: str, train: TrainConfig, final: bool
) -> tuple[Path, None] | tuple[Path, float]:
    groups_root = (
        train.log_root
        / name
        / train.group_name_str
        / 'rrl'
        / ('final' if final else train.ref_name_str)
    )
    if (root := try_get_best_exp_root(groups_root, final)) is not None:
        return root
    else:
        raise IatreionException(
            'No experiment root found for $dataset and groups "$groups".',
            dataset=name,
            groups=train.group_name_str,
        )


def get_rrl_file(exp_root: Path, train: TrainConfig) -> Path:
    return exp_root / ('rrl.tsv' if train.final else f'rrl_{train.ith_kfold}.tsv')


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
