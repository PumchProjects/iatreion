import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingDirectory

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
        return float(match.group('value'))
    return 0.0


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
        self.dataset.simple = True
        if not self.train.final:
            self.train.log_dir = (
                self.train.log_root
                / self.dataset.name_str
                / self.train.group_name_str
                / 'rrl_discrete'
                / self.train.ref_name_str
            )
            add_file_handler(self.train.log_dir / 'eval.log')

    def get_best_exp_root(self, groups_root: Path) -> Path | None:
        if not groups_root.is_dir():
            return None
        if self.train.final:
            return groups_root
        best_f1 = 0.0
        best_exp_root: Path | None = None
        for exp_root in groups_root.iterdir():
            f1 = get_avg_f1(exp_root)
            if f1 > best_f1:
                best_f1 = f1
                best_exp_root = exp_root
        return best_exp_root

    def get_best_exp_roots(self) -> list[Path]:
        exp_roots: list[Path] = []
        for name in self.dataset.names:
            groups_root = (
                self.thesaurus
                / name
                / self.train.group_name_str
                / 'rrl'
                / self.train.ref_name_str
            )
            if (root := self.get_best_exp_root(groups_root)) is not None:
                exp_roots.append(root)
            else:
                raise IatreionException(
                    'No experiment root found for $dataset and groups "$groups".',
                    dataset=name,
                    groups=self.train.group_name_str,
                )
        return exp_roots

    def get_rrl_file(self, exp_root: Path) -> Path:
        return exp_root / (
            'rrl.tsv' if self.train.final else f'rrl_{self.train.ith_kfold}.tsv'
        )
