from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import Parameter

from .model_base import ImportanceMethod
from .show_result_interpretability import ShowInterpretabilityConfig
from .train import TrainConfig


@Parameter(name='*')
@dataclass(kw_only=True)
class ShowImportanceConfig(ShowInterpretabilityConfig):
    importance_methods: Annotated[
        list[ImportanceMethod], Parameter(alias='-im', consume_multiple=True)
    ] = field(default_factory=lambda: ['permutation'])
    'Feature-importance methods to visualize.'

    importance_abs: Annotated[bool, Parameter(negative='--signed-importance')] = True
    'Use absolute importance values before aggregation. Disable for signed values.'

    importance_normalize: Annotated[
        bool, Parameter(negative='--no-importance-norm')
    ] = True
    'Normalize each fold importance vector to sum 1 before aggregation.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self._register(self.importance_methods)

    def make_config(
        self,
    ) -> Generator[tuple[TrainConfig, str, str, ImportanceMethod], None, None]:
        for train_config, i in self._make_config():
            yield (
                train_config,
                self.results[i],
                self.labels[i],
                self.importance_methods[i],
            )
