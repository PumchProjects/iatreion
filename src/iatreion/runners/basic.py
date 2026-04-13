from typing import override

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.trainers import DelicateTrainer, ModelTrainer

from .base import Runner


class BasicRunner(Runner):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        super().__init__(model_cls, config)
        self.model = model_cls(config)
        self.trainer = DelicateTrainer if config.delicate else ModelTrainer

    @override
    def run(self) -> None:
        try:
            self.trainer(self.model).train()
        finally:
            self.model.close()
            self.base_config.close_log_handler()
