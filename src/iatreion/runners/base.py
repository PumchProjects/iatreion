from abc import ABC, abstractmethod

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.models.naming import model_name_for


class Runner(ABC):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        self.model_cls = model_cls
        self.base_config = config

    @property
    def model_name(self) -> str:
        return model_name_for(self.model_cls)

    @abstractmethod
    def run(self) -> None: ...
