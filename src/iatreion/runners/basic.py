from typing import override

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.trainers import ModelTrainer

from .base import Runner
from .final_calibration import fit_final_fusion_artifact, publish_fusion_artifact

BASELINE_FINAL_MODEL_NAMES = {
    'RandomForestModel': 'random_forest',
    'XgboostModel': 'xgboost',
}


class BasicRunner(Runner):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        super().__init__(model_cls, config)
        self.model = model_cls(config)

    @property
    def _baseline_model_name(self) -> str | None:
        return BASELINE_FINAL_MODEL_NAMES.get(self.model_cls.__name__)

    def _fit_final_fusion_artifact(self) -> None:
        model_name = self._baseline_model_name
        train = self.base_config.train
        if (
            model_name is None
            or not train.final
            or train.num_class != 2
            or train.aggregate != 'calibrated-fusion'
        ):
            return

        source = fit_final_fusion_artifact(
            self.model_cls,
            self.base_config,
            model_name,
        )
        publish_fusion_artifact(source, self.base_config.train._log_dir)

    @override
    def run(self) -> None:
        try:
            self._fit_final_fusion_artifact()
            ModelTrainer(self.model).train()
        finally:
            self.model.close()
            self.base_config.close_log_handler()
