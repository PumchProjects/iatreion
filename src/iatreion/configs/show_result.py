from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import Directory

from .dataset import DatasetConfig
from .model_base import ModelConfig
from .show_base import ShowConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowResultConfig(ShowConfig):
    models: Annotated[list[str], Parameter(alias='-m', consume_multiple=True)]
    'Model names to show results for.'

    aggregates: Annotated[
        list[Literal['average', 'concat', 'stack']],
        Parameter(alias='-a', consume_multiple=True),
    ] = field(default_factory=lambda: ['stack'])
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

    reference: Annotated[str | None, Parameter(alias='-ref')] = None
    'Reference label for comparison.'

    metrics: Annotated[list[str], Parameter(alias='-met', consume_multiple=True)] = (
        field(default_factory=lambda: ['AUC', 'SEN', 'SPC'])
    )
    'Metrics to include in the analysis.'

    log_root: Directory = Path('logs')
    'Root directory for logs.'

    @staticmethod
    def _max_len(*lists: list) -> int:
        return max(len(lst) for lst in lists)

    @staticmethod
    def _pad(lst: list, target_len: int) -> list:
        match len(lst):
            case 1:
                return lst * target_len
            case n if n == target_len:
                return lst
            case n:
                raise ValueError(
                    f'List length {n} does not match target length {target_len}.'
                )

    def _pad_lists(self) -> None:
        max_len = self._max_len(
            self.models, self.aggregates, self.true_refs, self.results, self.labels
        )
        self.models = self._pad(self.models, max_len)
        self.aggregates = self._pad(self.aggregates, max_len)
        self.true_refs = self._pad(self.true_refs, max_len)
        self.results = self._pad(self.results, max_len)
        self.labels = self._pad(self.labels, max_len)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._pad_lists()

    def _make_config(self) -> ModelConfig:
        # HACK: Empty prefix
        dataset_config = DatasetConfig(prefix=Path(), names=self.names)
        train_config = TrainConfig(
            group_names=self.groups, log_root=self.log_root, _shuffle=False
        )
        return ModelConfig(dataset=dataset_config, train=train_config)

    def make_configs(self) -> Generator[tuple[TrainConfig, str, str], None, None]:
        config = self._make_config()
        for model_name, aggregate, true_ref, result_name, label in zip(
            self.models,
            self.aggregates,
            self.true_refs,
            self.results,
            self.labels,
            strict=True,
        ):
            config.train.aggregate = aggregate
            config.train.true_ref = true_ref
            config.train._log_dir = config.get_exp_root(model_name)
            yield config.train, result_name, label
