from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.train_utils import get_cv_fold_specs, get_data_names, read_data
from iatreion.train_utils.fusion import (
    get_published_fusion_artifact_path,
    get_run_fusion_artifact_path,
)
from iatreion.trainers import ModelTrainer
from iatreion.utils import apply_overrides


@dataclass(frozen=True)
class FinalCalibrationTarget:
    aggregate: str
    eval_names: list[str]
    folder_name: str


def get_final_calibration_target(config: ModelConfig) -> FinalCalibrationTarget:
    names = get_data_names(config.dataset, config.train)
    if names == ['all_concat']:
        return FinalCalibrationTarget(
            aggregate='concat',
            eval_names=[],
            folder_name='all_concat',
        )
    return FinalCalibrationTarget(
        aggregate='average',
        eval_names=names,
        folder_name=config.train.eval_name_str or config.dataset.name_str,
    )


def fit_final_fusion_artifact(
    model_cls: type[Model],
    base_config: ModelConfig,
    model_name: str,
    *,
    parameter_map: dict[str, dict[str, Any]] | None = None,
) -> Path:
    train = base_config.train
    _, _, ref_y, _ = read_data(base_config.dataset, train)
    fold_specs = get_cv_fold_specs(train.n_inner_splits, ref_y)
    target = get_final_calibration_target(base_config)
    config = apply_overrides(
        base_config,
        {
            'importance_methods': [],
            'study_name': None,
            'tune_config': None,
            'train.aggregate': target.aggregate,
            'train._eval_names': target.eval_names,
            'train.final': False,
            'train.log_root': train.log_root / 'final-calibration',
            'train.n_outer_splits': len(fold_specs),
        },
    )
    default_log = config.train._log_dir / 'train.log'
    config.register_log_dir(model_name, folder_name=target.folder_name)
    if (
        default_log != config.train._log_dir / 'train.log'
        and default_log.is_file()
        and default_log.stat().st_size == 0
    ):
        default_log.unlink()

    model: Model | None = None
    try:
        model = model_cls(config)
        ModelTrainer(
            model,
            fold_specs=fold_specs,
            parameter_map=parameter_map,
            calc_ci=False,
        ).train()
    finally:
        if model is not None:
            model.close()
        config.close_log_handler()

    return get_run_fusion_artifact_path(config.train._log_dir)


def publish_fusion_artifact(
    source: Path, target_root: Path, subset_names: list[str]
) -> Path:
    subset_target = get_published_fusion_artifact_path(target_root, subset_names)
    subset_target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, subset_target)
    return subset_target
