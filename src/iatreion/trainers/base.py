from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import closing
from itertools import groupby

from iatreion.configs import ModelConfig
from iatreion.train_utils import TrainStepContext, get_train_iterator
from iatreion.utils import logger, task

from .recorder import Recorder, TrainerReturn
from .utils import record_simple, record_weighted_and_stacking


class Trainer(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.dataset_config, self.train_config = config.dataset, config.train

    @abstractmethod
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn: ...

    @abstractmethod
    def train_final(self, ctx: TrainStepContext) -> None: ...

    def train(self) -> None:
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
                if self.train_config.aggregate != 'concat':
                    record_simple(outer_fold, simple_recorder, outer_recorders)
                if self.train_config.aggregate == 'stack':
                    record_weighted_and_stacking(
                        outer_fold,
                        weighted_recorder,
                        stacking_recorder,
                        inner_recorders,
                        outer_recorders,
                    )

        if not self.train_config.final:
            with task('Data:', len(outer_recorders)) as outer_advance:
                for name, outer_recorder in outer_recorders.items():
                    outer_recorder.finish().log(name)
                    outer_advance()
            if self.train_config.aggregate != 'concat':
                simple_recorder.finish().log('all_simple_average')
            if self.train_config.aggregate == 'stack':
                weighted_recorder.finish().log('all_weighted_average')
                stacking_recorder.finish().log('all_stacking')
