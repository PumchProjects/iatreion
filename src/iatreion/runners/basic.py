from shutil import copyfile
from typing import override

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.train_utils import get_cv_fold_specs, read_data
from iatreion.train_utils.fusion import FUSION_ARTIFACT_FILE
from iatreion.trainers import ModelTrainer
from iatreion.utils import apply_overrides

from .base import Runner

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

        _, _, ref_y, _ = read_data(self.base_config.dataset, train)
        fold_specs = get_cv_fold_specs(train.n_inner_splits, ref_y)
        config = apply_overrides(
            self.base_config,
            {
                'importance_methods': [],
                'train.aggregate': 'average',
                'train.final': False,
                'train.n_outer_splits': len(fold_specs),
            },
        )
        config.register_log_dir(model_name, folder_name='final_calibration')
        model: Model | None = None
        try:
            model = self.model_cls(config)
            ModelTrainer(model, fold_specs=fold_specs, calc_ci=False).train()
        finally:
            if model is not None:
                model.close()
            config.close_log_handler()

        source = config.train._log_dir / FUSION_ARTIFACT_FILE
        target = self.base_config.train._log_dir / FUSION_ARTIFACT_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)

    @override
    def run(self) -> None:
        try:
            self._fit_final_fusion_artifact()
            ModelTrainer(self.model).train()
        finally:
            self.model.close()
            self.base_config.close_log_handler()
