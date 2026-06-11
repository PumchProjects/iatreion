from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from iatreion.configs import BaselineEvalConfig, ModelConfig
from iatreion.exceptions import IatreionException
from iatreion.models import Model
from iatreion.models.base import get_final_artifact_dir, get_transform_artifact_path
from iatreion.models.naming import model_name_for
from iatreion.preprocessors import get_preprocessors
from iatreion.train_utils import make_data_labels
from iatreion.train_utils.fusion import (
    AvailableFusionArtifact,
    get_published_fusion_artifact_path,
)
from iatreion.train_utils.preprocessing import DBEncoderArtifact
from iatreion.trainers import Recorder, TrainerReturn
from iatreion.utils import write_spreadsheet


@dataclass(frozen=True)
class BaselinePredictionResult:
    result: pd.DataFrame
    additional_data: list[pd.DataFrame]
    group_names: pd.DataFrame | None
    model_config: ModelConfig
    artifact: AvailableFusionArtifact


def _combined_index(data: list[pd.DataFrame]) -> pd.Index:
    index = data[0].index
    for frame in data[1:]:
        index = index.union(frame.index, sort=False)
    return index


def _load_fusion_artifact(model_config: ModelConfig) -> AvailableFusionArtifact:
    artifact = AvailableFusionArtifact.load(
        get_published_fusion_artifact_path(
            model_config.train._log_dir,
            list(model_config.dataset.names),
        )
    )
    if artifact.labels != model_config.train.group_labels:
        raise IatreionException(
            'Available-fusion labels [$actual] do not match requested labels '
            '[$expected].',
            actual=', '.join(artifact.labels),
            expected=', '.join(model_config.train.group_labels),
        )
    if artifact.positive_label != model_config.train.positive_label:
        raise IatreionException(
            'Available-fusion positive label "$actual" does not match "$expected".',
            actual=artifact.positive_label,
            expected=model_config.train.positive_label,
        )
    return artifact


def _validate_names(names: list[str], artifact: AvailableFusionArtifact) -> None:
    missing = sorted(set(names) - set(artifact.names))
    if missing:
        raise IatreionException(
            'Available-fusion artifact does not contain module(s): $names.',
            names=', '.join(missing),
        )


def _predict_labels(
    result: pd.DataFrame, artifact: AvailableFusionArtifact
) -> pd.Series:
    y_pos_score = result[artifact.positive_label].to_numpy()
    labels = artifact.predict_labels(y_pos_score)
    labels[pd.isna(result).all(axis=1).to_numpy()] = ''
    return pd.Series(labels, index=result.index, name='Label')


def _get_external_data(
    config: BaselineEvalConfig,
    model_config_cls: type[ModelConfig],
    model_name: str,
) -> tuple[
    list[pd.DataFrame],
    list[pd.DataFrame],
    pd.DataFrame | None,
    ModelConfig,
]:
    process_config, model_config = config.make_configs(model_config_cls)
    model_config.train._log_dir = model_config.get_exp_root(model_name)
    preprocessors = get_preprocessors(process_config)
    data = [preprocessor.get_data_outer() for preprocessor in preprocessors]
    additional_data = process_config._final_indices
    group_names = preprocessors[0].get_group_names() if config.mode == 'eval' else None
    return data, additional_data, group_names, model_config


def _predict_module(
    model_cls: type[Model],
    model_config: ModelConfig,
    name: str,
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    artifact_dir = get_final_artifact_dir(model_config.train._log_dir, name)
    transform = DBEncoderArtifact.load(
        get_transform_artifact_path(model_config.train._log_dir, name)
    )
    if transform.missing_value_strategy == 'limix':
        raise IatreionException(
            'External baseline evaluation does not support LimiX imputation.'
        )
    X, mask = transform.transform(frame)
    model = model_cls(model_config)
    try:
        model.load_final(artifact_dir, transform)
        y_score = model.predict_proba(X)
    finally:
        model.close()

    y_pos_score = y_score[:, model_config.train.positive_index]
    return (
        pd.Series(y_pos_score, index=frame.index, dtype=float),
        pd.Series(mask, index=frame.index, dtype=bool),
    )


def get_baseline_prediction_result(
    config: BaselineEvalConfig,
    model_cls: type[Model],
    model_config_cls: type[ModelConfig],
) -> BaselinePredictionResult:
    data, additional_data, group_names, model_config = _get_external_data(
        config, model_config_cls, model_name_for(model_cls)
    )
    artifact = _load_fusion_artifact(model_config)
    names = list(model_config.dataset.names)
    _validate_names(names, artifact)

    index = _combined_index(data)
    y_pos_score_list = []
    y_mask_list = []
    for name, frame in zip(names, data, strict=True):
        y_pos_score, y_mask = _predict_module(model_cls, model_config, name, frame)
        y_pos_score_list.append(y_pos_score.reindex(index).fillna(0.5).to_numpy())
        y_mask_list.append(y_mask.reindex(index, fill_value=True).to_numpy())

    result = pd.DataFrame(
        artifact.predict_scores(names, y_pos_score_list, y_mask_list),
        index=index,
        columns=artifact.labels,
    )
    available_any = ~np.column_stack(y_mask_list).all(axis=1)
    result.loc[~available_any] = np.nan
    return BaselinePredictionResult(
        result,
        additional_data,
        group_names,
        model_config,
        artifact,
    )


def get_baseline_batched_result(
    config: BaselineEvalConfig,
    model_cls: type[Model],
    model_config_cls: type[ModelConfig],
) -> tuple[pd.DataFrame, ModelConfig]:
    prediction = get_baseline_prediction_result(config, model_cls, model_config_cls)
    y_pred = _predict_labels(prediction.result, prediction.artifact)
    probability = prediction.result.add_prefix('Probability ')
    table = pd.concat(prediction.additional_data + [y_pred, probability], axis=1)
    return table, prediction.model_config


def get_baseline_eval_result(
    config: BaselineEvalConfig,
    model_cls: type[Model],
    model_config_cls: type[ModelConfig],
) -> tuple[str, Figure | None, ModelConfig]:
    prediction = get_baseline_prediction_result(config, model_cls, model_config_cls)
    group_names = prediction.group_names
    assert group_names is not None
    result = pd.concat([prediction.result, group_names], axis=1)
    train_config = prediction.model_config.train
    X_df, y_df = make_data_labels(result, train_config, group_names.columns.to_list())
    available = ~X_df.isna().all(axis=1)
    X_df = X_df.loc[available]
    y_df = y_df.loc[available]
    y_true = y_df.map(train_config.get_group_index_mapping()).to_numpy()
    y_score = X_df.to_numpy()
    recorder = Recorder(train_config)
    eval_result = recorder.record(
        TrainerReturn(
            0.0,
            y_true,
            y_score,
            threshold=prediction.artifact.default_threshold,
        )
    )
    fig = recorder.roc.fig if train_config.plot_roc else None
    artifact_path = get_published_fusion_artifact_path(
        prediction.model_config.train._log_dir,
        list(prediction.model_config.dataset.names),
    )
    summary = f'Final calibrated-fusion baseline evaluation\nArtifact: {artifact_path}'
    return f'{summary}\n\n{eval_result}', fig, prediction.model_config


def save_baseline_batched_result_table(table: pd.DataFrame, path: str | Path) -> Path:
    return write_spreadsheet(path, table, float_format='%.4f')
