import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory

from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig

avg_f1_pattern = re.compile(r'AVG F1\s+(.+?)%')


def get_avg_f1(exp_root: Path) -> float:
    file = exp_root / 'train.log'
    if not file.exists():
        return 0.0
    data = file.read_text(encoding='utf-8')
    match = avg_f1_pattern.search(data)
    if match:
        return float(match.group(1))
    return 0.0


def get_best_exp_root(groups_root: Path) -> Path | None:
    best_f1 = 0.0
    best_exp_root: Path | None = None
    for exp_root in (groups_root / 'rrl').iterdir():
        f1 = get_avg_f1(exp_root)
        if f1 > best_f1:
            best_f1 = f1
            best_exp_root = exp_root
    return best_exp_root


@Parameter(name='*')
@dataclass(kw_only=True)
class DiscreteRrlConfig:
    dataset: DatasetConfig

    train: TrainConfig

    thesaurus: Annotated[ExistingDirectory, Parameter(name=['--thesaurus', '-t'])] = (
        Path('logs')
    )
    'Root directory for trained RRL models.'

    def __post_init__(self) -> None:
        if not self.train.final:
            self.train.log_dir = (
                self.train.log_root
                / self.dataset.name
                / self.train.group_names
                / 'rrl_discrete'
            )
            add_file_handler(self.train.log_dir / 'eval.log')

    def get_best_exp_root(self) -> Path | None:
        groups_root = self.thesaurus / self.dataset.name / self.train.group_names
        if self.train.final:
            return root if (root := groups_root / 'rrl' / 'final').is_dir() else None
        return get_best_exp_root(groups_root)

    def get_rrl_file(self, exp_root: Path) -> Path:
        if self.train.final:
            return exp_root / 'rrl.tsv'
        return exp_root / f'rrl_{self.train.ith_kfold}.tsv'
