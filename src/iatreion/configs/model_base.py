from dataclasses import dataclass, field
from functools import cached_property
from logging import FileHandler
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import ExistingFile

from iatreion.exceptions import IatreionException
from iatreion.utils import add_file_handler, remove_file_handler

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
    ] = field(default_factory=list)
    'Feature-importance methods to export. Available: native, permutation, shap.'

    importance_repeats: Annotated[int, Parameter(alias='-ir')] = 5
    'Number of repeats for permutation importance.'

    importance_max_samples: Annotated[int | None, Parameter(alias='-ims')] = 256
    'Maximum number of test samples used for permutation/SHAP importance. Disable with None.'

    study_name: Annotated[str | None, Parameter(alias='-sn')] = None
    'Optuna study name. If not provided, use the default name defined in the TOML file.'

    tune_config: ExistingFile | None = None
    'Path to the TOML file that defines the Optuna study and search space.'

    delicate_config: ExistingFile | None = None
    'Path to the TOML file that defines the delicate training procedure. Only used for RRL.'

    _log_handler: FileHandler | None = field(init=False, default=None, repr=False)

    _delicate_flag: bool = False

    @property
    def tune(self) -> bool:
        return self.tune_config is not None
    
    @property
    def delicate(self) -> bool:
        return self.delicate_config is not None

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
        if self.tune or self._delicate_flag:
            return
        self.train._log_dir = self.get_exp_root(model_name)
        if folder_name is not None and not self.train.final:
            self.train._log_dir /= folder_name
        self.close_log_handler()
        self._log_handler = add_file_handler(self.train._log_dir / file_name)

    def close_log_handler(self) -> None:
        if self._log_handler is None:
            return
        remove_file_handler(self._log_handler)
        self._log_handler = None
