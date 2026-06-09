from abc import ABC, abstractmethod

from iatreion.configs import ModelConfig
from iatreion.models import Model

MODEL_LOG_NAMES = {
    'DiscreteRrlModel': 'rrl-discrete',
    'LimiXModel': 'limix',
    'RandomForestModel': 'random_forest',
    'RrlModel': 'rrl',
    'TabPFNModel': 'tabpfn',
    'XgboostModel': 'xgboost',
}


def model_name_for(model_cls: type[Model]) -> str:
    name = model_cls.__name__
    if name in MODEL_LOG_NAMES:
        return MODEL_LOG_NAMES[name]
    name = name.removesuffix('Model')
    return name[:1].lower() + name[1:]


class Runner(ABC):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        self.model_cls = model_cls
        self.base_config = config

    @property
    def model_name(self) -> str:
        return model_name_for(self.model_cls)

    @abstractmethod
    def run(self) -> None: ...
