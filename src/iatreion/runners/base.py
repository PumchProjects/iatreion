from abc import ABC, abstractmethod

from iatreion.configs import ModelConfig
from iatreion.models import Model


class Runner(ABC):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        self.model_cls = model_cls
        self.base_config = config

    @abstractmethod
    def run(self) -> None: ...
