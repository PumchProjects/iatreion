from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

from .dataset import DatasetConfig
from .model_base import ModelConfig
from .show_base import ShowConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowResultConfig(ShowConfig):
    log_root: Directory = Path('logs')
    'Root directory for logs.'

    models: Annotated[list[str], Parameter(alias='-m', consume_multiple=True)]
    'Model names to show results for.'

    aggregates: Annotated[
        list[Literal['average', 'concat', 'stack']],
        Parameter(alias='-a', consume_multiple=True),
    ]
    """Aggregation strategy for multimodal samples of the same patient.
'average': simple average predictions of different modalities.
'concat': concatenate features of different modalities.
'stack': late fusion by stacking predictions of different modalities as features for a meta-classifier.
"""

    true_refs: Annotated[list[bool], Parameter(alias='-tr', consume_multiple=True)] = (
        field(default_factory=lambda: [False])
    )
    'Align not only the test data, but also the training data to the reference data.'

    results: Annotated[list[str], Parameter(alias='-r', consume_multiple=True)]
    'Result names to show.'

    labels: Annotated[list[str], Parameter(alias='-l', consume_multiple=True)]
    'Labels for the results to show.'

    _registered: list[list[Any]] = field(default_factory=list)

    def _register(self, *lists: list[Any]) -> None:
        self._registered += lists

    @staticmethod
    def _pad(lst: list[Any], target_len: int) -> None:
        match len(lst):
            case 1:
                lst *= target_len
            case n if n == target_len:
                pass
            case n:
                raise ValueError(
                    f'List length {n} does not match target length {target_len}.'
                )

    def _pad_lists(self) -> int:
        max_len = max(len(lst) for lst in self._registered)
        for lst in self._registered:
            self._pad(lst, max_len)
        return max_len

    def __post_init__(self) -> None:
        super().__post_init__()
        self._register(
            self.models, self.aggregates, self.true_refs, self.results, self.labels
        )

    def _make_config(self) -> Generator[tuple[TrainConfig, int], None, None]:
        # HACK: Empty prefix
        dataset_config = DatasetConfig(prefix=Path(), names=self.names)
        train_config = TrainConfig(
            group_names=self.groups, log_root=self.log_root, _shuffle=False
        )
        config = ModelConfig(dataset=dataset_config, train=train_config)
        pad_len = self._pad_lists()
        for i in range(pad_len):
            train_config.aggregate = self.aggregates[i]
            train_config.true_ref = self.true_refs[i]
            train_config._log_dir = config.get_exp_root(self.models[i])
            yield train_config, i

    def make_config(self) -> Generator[tuple[TrainConfig, str, str], None, None]:
        for train_config, i in self._make_config():
            yield train_config, self.results[i], self.labels[i]
