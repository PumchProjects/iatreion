from abc import ABC, abstractmethod

from iatreion.configs import ModelConfig
from iatreion.models import Model


class Runner(ABC):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        self.model_cls = model_cls
        self.base_config = config

    @property
    def model_name(self) -> str:
        name = self.model_cls.__name__.removesuffix('Model')
        return name[:1].lower() + name[1:]

    @abstractmethod
    def run(self) -> None: ...
