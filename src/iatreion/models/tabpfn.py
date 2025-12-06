import json
from typing import override

from numpy.typing import NDArray
from tabpfn import TabPFNClassifier

from iatreion.configs import TabPFNConfig
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
        self.feature_names = self.get_feature_names()

    def get_feature_names(self) -> list[str]:
        feature_names: list[str] = []
        for name in self.config.dataset.names:
            fmap = self.config.dataset.get_fmap(name)
            with fmap.open('r', encoding='utf-8') as f:
                feature_names += [decode_string(line.split('\t')[1]) for line in f]
        return feature_names

    @override
    def fit(self, X: NDArray, y: NDArray) -> None:
        self.model.fit(X, y)

    def calc_importance(self, X: NDArray) -> None:
        from tabpfn_extensions import interpretability

        shap_values = interpretability.shap.get_shap_values(
            estimator=self.model,
            test_x=X,
            attribute_names=self.feature_names,
            algorithm='permutation',
            max_evals=2 * len(self.feature_names) + 1,
            seed=0,
        )
        importances = shap_values[:, :, 0].abs.mean(0).values
        score = {
            name: importances[i].item() for i, name in enumerate(self.feature_names)
        }
        with self.config.score_file.open('w', encoding='utf-8') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)

    @override
    def predict(self, X: NDArray, y: NDArray) -> ModelReturn:
        y_score = self.model.predict_proba(X)
        if self.config.calc_importance:
            self.calc_importance(X)
        return y_score, {}
