import json
from typing import override

from numpy.typing import NDArray
from tabpfn import TabPFNClassifier

from iatreion.configs import TabPFNConfig
from iatreion.rrl import TrainStepContext
from iatreion.utils import decode_string

from .base import Model, ModelReturn


class TabPFNModel(Model):
    def __init__(self, config: TabPFNConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TabPFNClassifier(
            model_path=config.model_path,
            memory_saving_mode=False,
            random_state=0,
            n_preprocessing_jobs=config.n_jobs,
        )

    @override
    def fit(self, X: NDArray, y: NDArray) -> None:
        self.model.fit(X, y)

    def calc_importance(self, ctx: TrainStepContext, X: NDArray) -> None:
        from tabpfn_extensions import interpretability

        feature_names = [decode_string(name) for name in ctx.db_enc.X_fname]
        shap_values = interpretability.shap.get_shap_values(
            estimator=self.model,
            test_x=X,
            attribute_names=feature_names,
            algorithm='permutation',
            max_evals=2 * len(feature_names) + 1,
            seed=0,
        )
        importances = shap_values[:, :, 0].abs.mean(0).values
        score = {name: importances[i].item() for i, name in enumerate(feature_names)}
        score_file = (
            self.config.train.log_dir
            / f'{ctx.name}_{ctx.outer_fold}_{ctx.inner_fold}.json'
        )
        with score_file.open('w', encoding='utf-8') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)

    @override
    def predict(self, ctx: TrainStepContext, X: NDArray, y: NDArray) -> ModelReturn:
        y_score = self.model.predict_proba(X)
        if self.config.calc_importance:
            self.calc_importance(ctx, X)
        return y_score, {}
