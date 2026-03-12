from typing import Any, override

from numpy.typing import NDArray

from iatreion.configs import RrlConfig
from iatreion.rrl import TrainStepContext
from iatreion.rrl.experiment import (
    calc_complexity,
    print_rules,
    test_model,
    train_model,
)
from iatreion.rrl.rrl.models import RRL
from iatreion.utils import set_seed_torch

from .base import Model


class RrlModel(Model):
    def __init__(self, config: RrlConfig) -> None:
        super().__init__()
        self.config: RrlConfig = config
        self.model: RRL | None = None
        self.state_dict: dict[str, Any] = {}
        self.metrics: tuple[float, ...] = ()
        self.rule2weights: Any = None

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        raise NotImplementedError

    def save_model_callback(
        self, model: RRL, state_dict: dict[str, Any], metrics: tuple[float, ...]
    ) -> None:
        self.model = model
        self.state_dict = state_dict
        self.metrics = metrics

    def load_model(self) -> None:
        if self.model is not None and self.state_dict:
            self.model.net.load_state_dict(self.state_dict)

    @override
    def fit(self, ctx: TrainStepContext) -> None:
        # HACK: Reset the seed for each training step to ensure reproducibility
        set_seed_torch(self.config.train.seed)
        train_model(self.config, self.save_model_callback, ctx)
        self.load_model()
        self.rule2weights = print_rules(self.config, ctx, self.model, self.metrics)

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        assert self.model is not None
        return test_model(self.config, X, self.model)

    @override
    def _calc_complexity(self) -> dict[str, float | tuple[float, str]]:
        assert self.model is not None
        return {'Log#E': calc_complexity(self.model, self.rule2weights)}
