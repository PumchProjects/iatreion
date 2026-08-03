from dataclasses import dataclass, field
from functools import cached_property
from logging import FileHandler
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import ExistingTomlPath

from iatreion.exceptions import IatreionException
from iatreion.log_paths import (
    EVAL_DIR,
    FINAL_DIR,
    final_model_root,
    training_model_root,
)
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

    fold_scope: FoldScope = 'outer'
    """Fold scope for importance calculation.
'outer': only calculate importance for outer folds.
'all': also calculate importance for inner folds.
"""

    importance_methods: Annotated[
        list[ImportanceMethod], Parameter(consume_multiple=True)
    ] = field(default_factory=list)
    """Feature-importance methods to export. Available: native, permutation, shap.
Final fitting exports native importance only.
"""

    importance_repeats: int = 5
    'Number of repeats for permutation importance.'

    importance_max_samples: int | None = 256
    'Maximum number of test samples used for permutation/SHAP importance. Disable with None.'

    study_name: str | None = None
    'Optuna study name. If not provided, use the TOML name or an auto-generated dataset/group/aggregate name.'

    tune_config: ExistingTomlPath | None = None
    'Path to the TOML file that defines nested/final Optuna tuning.'

    _log_handler: FileHandler | None = field(init=False, default=None, repr=False)

    @property
    def tune(self) -> bool:
        return self.tune_config is not None

    @property
    def log_folder_name(self) -> str | None:
        return None

    def get_internal_exp_root(self, model_name: str) -> Path:
        return training_model_root(
            self.train.log_root,
            self.dataset.name_str,
            self.train.group_name_str,
            model_name,
            self.train.ref_name_str,
        )

    def get_final_exp_root(self, model_name: str) -> Path:
        return final_model_root(
            self.train.log_root,
            self.train.group_name_str,
            model_name,
        )

    def get_exp_root(self, model_name: str) -> Path:
        if self.train.final:
            return self.get_final_exp_root(model_name)
        return self.get_internal_exp_root(model_name)

    def get_eval_root(self, model_name: str) -> Path:
        return self.get_exp_root(model_name) / EVAL_DIR / self.dataset.name_str

    @cached_property
    def rrl_root(self) -> Path:
        exp_root = self.get_exp_root('rrl')
        if not exp_root.is_dir():
            raise IatreionException(
                'No experiment root found for $dataset and groups "$groups".',
                dataset=FINAL_DIR if self.train.final else self.dataset.name_str,
                groups=self.train.group_name_str,
            )
        return exp_root

    def register_log_dir(
        self,
        model_name: str,
        *,
        root: Path | None = None,
        folder_name: str | None = None,
        file_name: str = 'train.log',
    ) -> None:
        if self.tune:
            return
        self.train._log_dir = (
            root if root is not None else self.get_exp_root(model_name)
        )
        if folder_name is not None:
            self.train._log_dir /= folder_name
        self.close_log_handler()
        self._log_handler = add_file_handler(self.train._log_dir / file_name)

    def register_eval_log_dir(self, model_name: str) -> None:
        if self.tune:
            return
        self.train._log_dir = self.get_eval_root(model_name)
        self.close_log_handler()
        self._log_handler = add_file_handler(self.train._log_dir / 'eval.log')

    def close_log_handler(self) -> None:
        if self._log_handler is None:
            return
        remove_file_handler(self._log_handler)
        self._log_handler = None
