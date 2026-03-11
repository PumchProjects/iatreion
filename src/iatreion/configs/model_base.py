from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

from iatreion.exceptions import IatreionException
from iatreion.utils import add_file_handler

from .dataset import DatasetConfig
from .train import TrainConfig

type FoldScope = Literal['outer', 'all']
type ImportanceMethod = Literal['native', 'permutation', 'shap']


@Parameter(name='*')
@dataclass(kw_only=True)
class ModelConfig:
    dataset: DatasetConfig
    train: TrainConfig

    fold_scope: Annotated[FoldScope, Parameter(alias='-is')] = 'outer'
    """Fold scope for importance calculation.
'outer': only calculate importance for outer folds.
'all': also calculate importance for inner folds.
"""

    importance_methods: Annotated[
        list[ImportanceMethod], Parameter(alias='-im', consume_multiple=True)
    ] = field(default_factory=lambda: ['permutation'])
    'Feature-importance methods to export. Available: native, permutation, shap.'

    importance_repeats: Annotated[int, Parameter(alias='-ir')] = 5
    'Number of repeats for permutation importance.'

    importance_max_samples: Annotated[int | None, Parameter(alias='-ims')] = 256
    'Maximum number of test samples used for permutation/SHAP importance. Disable with None.'

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
