from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import groupby

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.rrl import TrainStepContext, get_train_iterator
from iatreion.utils import logger, progress, task

from .recorder import Recorder, TrainerReturn
from .utils import record_simple, record_stacking, record_weighted


class Trainer(ABC):
    def __init__(
        self, dataset_config: DatasetConfig, train_config: TrainConfig
    ) -> None:
        self.dataset_config = dataset_config
        self.train_config = train_config

    @abstractmethod
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn: ...

    @abstractmethod
    def train_final(self) -> None: ...

    def train(self) -> None:
        iterator = get_train_iterator(self.dataset_config, self.train_config)
        simple_recorder = Recorder(self.train_config)
        weighted_recorder = Recorder(self.train_config)
        stacking_recorder = Recorder(self.train_config)
        outer_recorders = defaultdict(lambda: Recorder(self.train_config))
        with progress, task('Fold:', self.train_config.n_folds) as fold_advance:
            for outer_fold, outer_group in groupby(
                iterator, lambda ctx: ctx.outer_fold
            ):
                inner_recorders = defaultdict(lambda: Recorder(self.train_config))
                for _, inner_group in groupby(outer_group, lambda ctx: ctx.inner_fold):
                    with task(
                        'Data:',
                        len(self.dataset_config.names),
                        self.train_config.aggregate != 'concat',
                    ) as data_advance:
                        for ctx in inner_group:
                            results = self.train_step(ctx)
                            if ctx.last:
                                logger.info(outer_recorders[ctx.name].record(results))
                            else:
                                logger.info(inner_recorders[ctx.name].record(results))
                            data_advance()
                    fold_advance()
                if self.train_config.aggregate != 'concat':
                    record_simple(simple_recorder, outer_recorders)
                if self.train_config.aggregate == 'stack':
                    record_weighted(
                        outer_fold, weighted_recorder, inner_recorders, outer_recorders
                    )
                    record_stacking(
                        outer_fold, stacking_recorder, inner_recorders, outer_recorders
                    )
        for name, outer_recorder in outer_recorders.items():
            outer_recorder.finish().log(name)
        if self.train_config.aggregate != 'concat':
            simple_recorder.finish().log('all_simple_average')
        if self.train_config.aggregate == 'stack':
            weighted_recorder.finish().log('all_weighted_average')
            stacking_recorder.finish().log('all_stacking')
