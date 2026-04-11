from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from itertools import groupby

from iatreion.configs import ModelConfig
from iatreion.train_utils import TrainStepContext, get_train_iterator
from iatreion.utils import logger, task

from .recorder import Finish, Recorder, TrainerReturn
from .utils import record_all, record_simple


@dataclass(frozen=True)
class TrainerSummary:
    finishes: dict[str, Finish] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)


class Trainer(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.dataset_config, self.train_config = config.dataset, config.train
        self.finishes: dict[str, Finish] = {}
        self.objectives: dict[str, float] = {}

    @abstractmethod
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn: ...

    @abstractmethod
    def train_final(self, ctx: TrainStepContext) -> None: ...

    def _store_finish(self, name: str, recorder: Recorder) -> None:
        finish = recorder.finish()
        finish.log(name)
        self.finishes[name] = finish
        for metric, value in finish.final.metrics.items():
            self.objectives[f'{name}/{metric}'] = value
        self.objectives[f'{name}/Time'] = finish.final.time
        for key, (value, _fmt) in finish.final.complexity.items():
            self.objectives[f'{name}/{key}'] = value
        if finish.ci is None:
            return
        for metric, (_point, lower, upper) in finish.ci.items():
            self.objectives[f'{name}/{metric}_lb'] = lower
            self.objectives[f'{name}/{metric}_ub'] = upper

    def train(self) -> TrainerSummary:
        iterator = get_train_iterator(self.dataset_config, self.train_config)

        simple_recorder = Recorder(self.train_config)
        weighted_recorder = Recorder(self.train_config)
        stacking_recorder = Recorder(self.train_config)
        outer_recorders = defaultdict(lambda: Recorder(self.train_config))

        with (
            closing(iterator),
            task(
                'Fold:', self.train_config.n_folds, not self.train_config.final
            ) as fold_advance,
        ):
            for outer_fold, outer_group in groupby(
                iterator, lambda ctx: ctx.outer_fold
            ):
                inner_recorders = defaultdict(
                    lambda: Recorder(self.train_config, is_inner=True)
                )

                for _, inner_group in groupby(outer_group, lambda ctx: ctx.inner_fold):
                    with task(
                        'Data:',
                        len(self.dataset_config.names),
                        self.train_config.aggregate != 'concat',
                    ) as data_advance:
                        for ctx in inner_group:
                            if self.train_config.final:
                                self.train_final(ctx)
                                data_advance()
                                continue

                            results = self.train_step(ctx)
                            if ctx.is_inner:
                                logger.info(inner_recorders[ctx.name].record(results))
                            else:
                                logger.info(outer_recorders[ctx.name].record(results))

                            data_advance()

                    fold_advance()

                if self.train_config.final:
                    continue
                if self.train_config.aggregate == 'stack':
                    record_all(
                        outer_fold,
                        simple_recorder,
                        weighted_recorder,
                        stacking_recorder,
                        inner_recorders,
                        outer_recorders,
                    )
                elif self.train_config.aggregate != 'concat':
                    record_simple(outer_fold, simple_recorder, outer_recorders)

        if not self.train_config.final:
            with task('Data:', len(outer_recorders)) as outer_advance:
                for name, outer_recorder in outer_recorders.items():
                    self._store_finish(name, outer_recorder)
                    outer_advance()
            if self.train_config.aggregate != 'concat':
                self._store_finish('all_simple_average', simple_recorder)
            if self.train_config.aggregate == 'stack':
                self._store_finish('all_weighted_average', weighted_recorder)
                self._store_finish('all_stacking', stacking_recorder)

        return TrainerSummary(
            finishes=self.finishes,
            objectives=self.objectives,
        )
