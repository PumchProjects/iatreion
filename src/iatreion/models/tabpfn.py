from typing import override

from numpy.typing import NDArray
from tabpfn import TabPFNClassifier

from iatreion.configs import TabPFNConfig

from .base import Model


class TabPFNModel(Model):
    def __init__(self, config: TabPFNConfig) -> None:
        super().__init__()
        self.config: TabPFNConfig = config
        self.model = TabPFNClassifier(
            model_path=config.model_path,
            memory_saving_mode=False,
            random_state=0,
            n_preprocessing_jobs=config.n_jobs,
        )

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self.model.fit(X, y)

    @override
    def _predict_proba(self, X: NDArray, y: NDArray) -> NDArray:
        return self.model.predict_proba(X)
