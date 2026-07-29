from pathlib import Path
from typing import override

from numpy.typing import NDArray
from tabpfn import TabPFNClassifier

from iatreion.configs import TabPFNConfig
from iatreion.train_utils import TrainStepContext
from iatreion.train_utils.artifacts import (
    get_artifact_dir,
    get_transform_artifact_path,
)
from iatreion.train_utils.preprocessing import DBEncoderArtifact

from .base import Model

TABPFN_MODEL_FILE = 'model.tabpfn_fit'


class TabPFNModel(Model):
    def __init__(self, config: TabPFNConfig) -> None:
        super().__init__()
        self.config: TabPFNConfig = config
        self.model = TabPFNClassifier(
            model_path=str(config.model_path),
            memory_saving_mode=False,
            random_state=0,
            n_preprocessing_jobs=config.n_jobs,
        )

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        self.model.fit(X, y)

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        return self.model.predict_proba(X)

    @override
    def save_final(self, ctx: TrainStepContext) -> None:
        artifact_dir = get_artifact_dir(self.config.train._log_dir, ctx.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ctx.db_enc.save_transform_artifact(
            get_transform_artifact_path(self.config.train._log_dir, ctx.name)
        )
        self.model.save_fit_state(artifact_dir / TABPFN_MODEL_FILE)

    @override
    def load_final(self, artifact_dir: Path, transform: DBEncoderArtifact) -> None:
        self.model = TabPFNClassifier.load_from_fit_state(
            artifact_dir / TABPFN_MODEL_FILE,
            device='auto',
        )
