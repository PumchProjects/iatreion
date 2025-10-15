import json
from typing import override

from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from iatreion.configs import RandomForestConfig
from iatreion.utils import decode_string

from .base import Model, ModelReturn


class RandomForestModel(Model):
    def __init__(self, config: RandomForestConfig) -> None:
        super().__init__()
        self.config = config
        self.num_class = config.train.num_class
        self.forest = RandomForestClassifier(
            config.n_estimators,
            n_jobs=config.n_jobs,
            random_state=0,
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
        self.forest.fit(X, y)
        importances = self.forest.feature_importances_
        score = {
            name: importances[i].item() for i, name in enumerate(self.feature_names)
        }
        with self.config.score_file.open('w', encoding='utf-8') as f:
            json.dump(score, f, ensure_ascii=False, indent=4)

    @override
    def predict(self, X: NDArray, y: NDArray) -> ModelReturn:
        y_score = self.forest.predict_proba(X)
        return y_score, {}
