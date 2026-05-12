from typing import override

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.trainers import ModelTrainer

from .base import Runner


class BasicRunner(Runner):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        super().__init__(model_cls, config)
        self.model = model_cls(config)

    @override
    def run(self) -> None:
        try:
            ModelTrainer(self.model).train()
        finally:
            self.model.close()
            self.base_config.close_log_handler()
