from abc import ABC, abstractmethod
from collections.abc import Callable

from numpy.typing import NDArray

from iatreion.configs import ImportanceMethod, ModelConfig
from iatreion.rrl import TrainStepContext
from iatreion.utils import logger

from .importance import (
    ImportanceScore,
    calc_permutation_importance,
    save_importance_score,
)

type ModelReturn = tuple[NDArray, dict[str, float | tuple[float, str]]]
type ImportanceCalculator = Callable[[TrainStepContext], ImportanceScore]


class Model(ABC):
    config: ModelConfig

    @abstractmethod
    def _fit(self, X: NDArray, y: NDArray) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: NDArray, y: NDArray) -> NDArray: ...

    def _calc_native_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        raise NotImplementedError

    def _calc_permutation_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        return calc_permutation_importance(self.config, ctx, self._predict_proba)

    def _calc_shap_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        raise NotImplementedError

    @property
    def _importance_calculators(self) -> dict[ImportanceMethod, ImportanceCalculator]:
        return {
            'native': self._calc_native_importance,
            'permutation': self._calc_permutation_importance,
            'shap': self._calc_shap_importance,
        }

    def _calc_importance(self, ctx: TrainStepContext) -> None:
        if self.config.importance_scope == 'outer' and ctx.is_inner:
            return

        for method in self.config.importance_methods:
            calculator = self._importance_calculators.get(method)
            if calculator is None:
                continue
            try:
                score = calculator(ctx)
            except Exception as error:
                logger.warning(
                    f'[bold yellow]Skip {method} importance for {ctx.name} '
                    f'(outer={ctx.outer_fold}, inner={ctx.inner_fold}): {error}',
                    extra={'markup': True},
                )
                continue
            save_importance_score(self.config.train, ctx, score, method=method)

    def _calc_complexity(self) -> dict[str, float | tuple[float, str]]:
        return {}

    def fit(self, ctx: TrainStepContext) -> None:
        self._fit(*ctx.train_data)

    def predict(self, ctx: TrainStepContext) -> ModelReturn:
        y_score = self._predict_proba(*ctx.test_data)
        self._calc_importance(ctx)
        return y_score, self._calc_complexity()
