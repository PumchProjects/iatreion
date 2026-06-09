from typing import Any, override

from iatreion.models import Model
from iatreion.train_utils import FoldSpec, TrainStepContext
from iatreion.train_utils.fusion import AvailableFusionArtifact
from iatreion.utils import Timer, apply_overrides

from .base import Trainer, TrainerReturn

type ParameterMap = dict[Any, Any]


class ModelTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        *,
        fold_specs: list[FoldSpec] | None = None,
        parameter_map: ParameterMap | None = None,
        calc_ci: bool = True,
    ) -> None:
        super().__init__(model.config, fold_specs=fold_specs, calc_ci=calc_ci)
        self.model = model
        self.base_config = model.config
        self.parameter_map = parameter_map or {}

    def _uses_available_fusion_artifact(self) -> bool:
        return (
            super()._uses_available_fusion_artifact()
            and not self.model.reuses_fusion_artifacts
        )

    def get_fusion_artifact(self, outer_fold: int) -> AvailableFusionArtifact | None:
        return self.model.get_fusion_artifact(outer_fold)

    def _get_overrides(self, ctx: TrainStepContext) -> dict[str, Any]:
        outer_params = self.parameter_map.get(ctx.outer_fold)
        if isinstance(outer_params, dict):
            name_params = outer_params.get(ctx.name)
            if isinstance(name_params, dict):
                return name_params

        name_params = self.parameter_map.get(ctx.name)
        if isinstance(name_params, dict):
            return name_params
        return {}

    def _update_config(self, ctx: TrainStepContext) -> None:
        overrides = self._get_overrides(ctx)
        if not overrides:
            self.model.config = self.base_config
            return

        log_dir = self.train_config._log_dir
        config = apply_overrides(self.base_config, overrides)
        config.train._log_dir = log_dir
        self.train_config._log_dir = log_dir
        self.model.config = config

    @override
    def train_step(self, ctx: TrainStepContext) -> TrainerReturn:
        # HACK: Validation set is not used for other models
        self._update_config(ctx)
        with Timer() as timer:
            self.model.fit(ctx)
        y_score, complexity = self.model.predict(ctx)
        return TrainerReturn(
            timer.duration,
            ctx.test_data[1],
            y_score,
            complexity,
            sample_id=ctx.test_index.astype(str).to_numpy(),
            outer_fold=ctx.outer_fold,
            inner_fold=ctx.inner_fold,
            kind='inner' if ctx.is_inner else 'outer',
            test_mask=ctx.test_mask,
        )

    @override
    def train_final(self, ctx: TrainStepContext) -> None:
        # HACK: Validation set is not used for other models
        self._update_config(ctx)
        self.model.fit(ctx)
        self.model.save_final(ctx)
