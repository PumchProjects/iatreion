from typing import override

from numpy.typing import NDArray
from tabpfn import TabPFNClassifier

from iatreion.configs import TabPFNConfig
from iatreion.rrl import TrainStepContext

from .base import Model
from .importance import ImportanceScore, sample_importance_data


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

    @override
    def _calc_shap_importance(self, ctx: TrainStepContext) -> ImportanceScore:
        from tabpfn_extensions import interpretability

        X_sample, _ = sample_importance_data(
            *ctx.test_data,
            max_samples=self.config.importance_max_samples,
            seed=self.config.train.seed,
        )
        feature_names = ctx.db_enc.X_fname
        shap_values = interpretability.shap.get_shap_values(
            estimator=self.model,
            test_x=X_sample,
            attribute_names=feature_names,
            algorithm='permutation',
            max_evals=2 * len(feature_names) + 1,
            seed=0,
        )
        importances = shap_values[:, :, 0].abs.mean(0).values
        return {name: float(importances[i]) for i, name in enumerate(feature_names)}
