from dataclasses import replace
from typing import override

from iatreion.models import Model
from iatreion.train_utils import TrainStepContext
from iatreion.utils import load_dict, logger

from .base import TrainerReturn
from .model import ModelTrainer


class DelicateTrainer(ModelTrainer):
    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.base_config = model.config
        self.base_config._delicate_flag = True
        assert self.base_config.delicate_config is not None
        self.delicate_configs = load_dict(self.base_config.delicate_config)
        self.check_names()

    def check_names(self) -> None:
        names = set(self.base_config.dataset.names) - set(self.delicate_configs.keys())
        if names:
            logger.warning(
                'The following data names do not have delicate configs '
                f'and will use the base config: {", ".join(names)}'
            )

    def update_config(self, name: str) -> None:
        # HACK: Configs read by model.__init__ are not updated
        if (delicate_config := self.delicate_configs.get(name)) is not None:
            self.model.config = replace(self.base_config, **delicate_config)
        else:
            self.model.config = self.base_config

    @override
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn:
        self.update_config(ctx.name)
        return super().train_step(ctx)

    @override
    def train_final(self, ctx: TrainStepContext) -> None:
        self.update_config(ctx.name)
        return super().train_final(ctx)
