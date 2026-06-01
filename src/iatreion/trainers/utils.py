from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

import numpy as np
from numpy.typing import NDArray

from iatreion.configs import DatasetConfig, TrainConfig
from iatreion.exceptions import IatreionException
from iatreion.train_utils.fusion import (
    FUSION_ARTIFACT_FILE,
    AvailableFusionArtifact,
    get_operating_thresholds,
)
from iatreion.utils import logger

from .recorder import PredictionRecord, Recorder, TrainerReturn, get_display_name


@dataclass
class LastPredictions:
    y_true: NDArray
    y_score_list: list[NDArray]
    y_mask_list: list[NDArray]
    names: list[str]
    time: float


def get_last_predictions(named_recorders: dict[str, Recorder]) -> LastPredictions:
    names = []
    time_list = []
    y_score_list = []
    y_mask_list = []
    for name, child in named_recorders.items():
        names.append(name)
        time_list.append(child.result.time[-1])
        y_score_list.append(child.result.y_all[-1].score)
        y_mask_list.append(child.result.y_all[-1].mask)
    time = sum(time_list)
    y_true = child.result.y_all[-1].true
    return LastPredictions(y_true, y_score_list, y_mask_list, names, time)


@dataclass
class FinalPredictions:
    y_true: NDArray
    y_pos_score_list: list[NDArray]
    y_mask_list: list[NDArray]
    names: list[str]


def get_final_predictions(
    fold: int, named_recorders: dict[str, Recorder]
) -> FinalPredictions:
    names = []
    y_pos_score_list = []
    y_mask_list = []
    for name, child in named_recorders.items():
        names.append(name)
        finish = child.finish(calc_ci=False)
        # HACK: Binary classification only
        y_pos_score_list.append(finish.final.y.score[:, 1])
        y_mask_list.append(finish.final.y.mask)
        finish.log(f'{name}_inner_{fold}')
    y_true = finish.final.y.true
    return FinalPredictions(y_true, y_pos_score_list, y_mask_list, names)


def log_available_fusion_artifact(
    config: TrainConfig, log_name: str, artifact: AvailableFusionArtifact
) -> None:
    width = max(len(name) for name in artifact.names)
    with config.logging(log_name):
        logger.info('Calibrated available-modality fusion')
        logger.info(
            f'Fusion policy: {artifact.fusion_policy} '
            f'(objective: {artifact.weight_objective}, '
            f'schema: {artifact.fusion_schema_version})'
        )
        for name in artifact.names:
            calibrator = artifact.calibrators[name]
            logger.info(
                f'Weight for {f"{name}:":{width + 1}} {artifact.weights[name]:.4f} '
                f'(Calibration: slope={calibrator.slope:.4f}, '
                f'intercept={calibrator.intercept:.4f})'
            )
        logger.info('Missing modalities are omitted and weights are renormalized.')
        if config.use_clinical_threshold:
            logger.info(
                f'Clinical recall threshold: '
                f'{artifact.thresholds["clinical_recall"]:.4f} '
                f'for recall({config.clinical_threshold_label}) '
                f'>= {config.clinical_threshold_recall:.4f}.'
            )
        logger.info(f'Youden threshold: {artifact.thresholds["youden"]:.4f}.')
        logger.info(
            f'Default threshold: {artifact.default_threshold_name} '
            f'({artifact.default_threshold:.4f}).'
        )


def fit_available_fusion_artifact(
    config: TrainConfig,
    fold: int,
    final: FinalPredictions,
    *,
    log_prefix: str = 'weights_available_fusion',
) -> AvailableFusionArtifact:
    artifact = AvailableFusionArtifact.fit(
        config,
        final.names,
        final.y_true,
        final.y_pos_score_list,
        final.y_mask_list,
    )
    log_available_fusion_artifact(config, f'{log_prefix}_{fold}', artifact)
    return artifact


def get_oof_predictions(named_recorders: dict[str, Recorder]) -> FinalPredictions:
    names = []
    y_pos_score_list = []
    y_mask_list = []
    for name, child in named_recorders.items():
        names.append(name)
        record = PredictionRecord.from_list(child.result.y_all)
        y_pos_score_list.append(record.score[:, 1])
        y_mask_list.append(record.mask)
    y_true = record.true
    return FinalPredictions(y_true, y_pos_score_list, y_mask_list, names)


def save_available_fusion_artifact(
    config: TrainConfig, named_recorders: dict[str, Recorder]
) -> None:
    final = get_oof_predictions(named_recorders)
    artifact = AvailableFusionArtifact.fit(
        config,
        final.names,
        final.y_true,
        final.y_pos_score_list,
        final.y_mask_list,
    )
    artifact.save(config._log_dir / FUSION_ARTIFACT_FILE)
    log_available_fusion_artifact(config, 'weights_available_fusion_artifact', artifact)


def get_thresholds(
    config: TrainConfig,
    y_true: NDArray,
    y_pos_score: NDArray,
    *,
    y_mask: NDArray | None = None,
) -> dict[str, float]:
    if y_mask is not None:
        observed = ~y_mask.astype(bool)
        y_true = y_true[observed]
        y_pos_score = y_pos_score[observed]
    return get_operating_thresholds(config, y_true, y_pos_score)


def get_all_missing_mask(y_mask_list: list[NDArray]) -> NDArray:
    return np.column_stack(y_mask_list).astype(bool).all(axis=1)


def average_available_scores(
    y_score_list: list[NDArray], y_mask_list: list[NDArray]
) -> NDArray:
    scores = np.stack(y_score_list)
    available = ~np.column_stack(y_mask_list).astype(bool).T
    numerator = (scores * available[:, :, None]).sum(axis=0)
    denominator = available.sum(axis=0)[:, None]
    average = np.full_like(numerator, 1.0 / numerator.shape[1], dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=average,
        where=denominator > 0,
    )


def aggregate_pos_scores(
    final: FinalPredictions,
    *,
    fusion_artifact: AvailableFusionArtifact | None = None,
) -> NDArray:
    if fusion_artifact is not None:
        y_pos_score = fusion_artifact.predict_pos_score(
            final.names, final.y_pos_score_list, final.y_mask_list
        )
    else:
        y_pos_score = average_available_scores(
            [np.column_stack([1 - score, score]) for score in final.y_pos_score_list],
            final.y_mask_list,
        )[:, 1]
    return y_pos_score


def aggregate_scores(
    last: LastPredictions,
    *,
    fusion_artifact: AvailableFusionArtifact | None = None,
) -> tuple[NDArray, list[float], float]:
    if fusion_artifact is not None:
        y_score = fusion_artifact.predict_scores(
            last.names,
            [score[:, 1] for score in last.y_score_list],
            last.y_mask_list,
        )
        norm_weights = [fusion_artifact.weights[name] for name in last.names]
        bias = 0.0
    else:
        y_score = average_available_scores(last.y_score_list, last.y_mask_list)
        n_total = len(last.y_score_list)
        norm_weights = np.full(n_total, 1 / n_total).tolist()
        bias = 0.0
    return y_score, norm_weights, bias


def aggregate(
    config: TrainConfig,
    fold: int,
    recorders: dict[str, Recorder],
    name: str,
    last: LastPredictions,
    final: FinalPredictions | None = None,
    *,
    fusion_artifact: AvailableFusionArtifact | None = None,
) -> None:
    y_score, norm_weights, bias = aggregate_scores(
        last, fusion_artifact=fusion_artifact
    )
    thresholds: dict[str, float | None] = {'original': None}
    if final is not None:
        final_mask = get_all_missing_mask(final.y_mask_list)
        y_pos_score = aggregate_pos_scores(final, fusion_artifact=fusion_artifact)
        thresholds |= get_thresholds(
            config, final.y_true, y_pos_score, y_mask=final_mask
        )
    last_mask = get_all_missing_mask(last.y_mask_list)
    for threshold_name, threshold in thresholds.items():
        recorder_name = f'{name}_{threshold_name}'
        recorder = recorders[recorder_name]
        recorder.record_weights_and_bias(norm_weights, bias)
        results = TrainerReturn(
            last.time,
            last.y_true,
            y_score,
            threshold=threshold,
            test_mask=last_mask,
        )
        display_name = get_display_name(recorder_name)
        logger.info(
            f'[bold green]{display_name} (Fold {fold}):', extra={'markup': True}
        )
        logger.info(recorder.record(results))


def record_average(
    config: TrainConfig,
    fold: int,
    recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    last = get_last_predictions(outer_recorders)
    aggregate(config, fold, recorders, 'all_simple_average', last)


def record_calibrated_concat(
    config: TrainConfig,
    fold: int,
    recorders: dict[str, Recorder],
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    last = get_last_predictions(outer_recorders)
    final = get_final_predictions(fold, inner_recorders)
    fusion_artifact = fit_available_fusion_artifact(
        config, fold, final, log_prefix='weights_calibrated_concat'
    )
    aggregate(
        config,
        fold,
        recorders,
        'all_calibrated_concat',
        last,
        final,
        fusion_artifact=fusion_artifact,
    )


def record_calibrated_fusion(
    config: TrainConfig,
    fold: int,
    recorders: dict[str, Recorder],
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    last = get_last_predictions(outer_recorders)
    final = get_final_predictions(fold, inner_recorders)

    fusion_artifact = fit_available_fusion_artifact(
        config, fold, final, log_prefix='weights_calibrated_fusion'
    )
    aggregate(
        config,
        fold,
        recorders,
        'all_calibrated_fusion',
        last,
        final,
        fusion_artifact=fusion_artifact,
    )


def get_final_available_fusion_artifact_source(
    dataset: DatasetConfig, train: TrainConfig
) -> Path:
    source = (
        train.log_root
        / dataset.name_str
        / train.group_name_str
        / 'rrl-discrete'
        / train.ref_name_str
    )
    if train.eval_names:
        source /= train.eval_name_str
    return source / FUSION_ARTIFACT_FILE


def validate_final_available_fusion_artifact(
    dataset: DatasetConfig, train: TrainConfig
) -> None:
    source = get_final_available_fusion_artifact_source(dataset, train)
    if source.is_file():
        return
    raise IatreionException(
        'Available-fusion artifact not found: $path. '
        'Run internal discrete RRL scoring before final RRL training.',
        path=str(source),
    )


def publish_final_available_fusion_artifact(
    dataset: DatasetConfig, train: TrainConfig
) -> None:
    source = get_final_available_fusion_artifact_source(dataset, train)
    target = train._log_dir / FUSION_ARTIFACT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, target)
    logger.info(f'Published available-fusion artifact: {target}')
