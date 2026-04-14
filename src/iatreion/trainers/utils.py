from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_curve

from iatreion.configs import TrainConfig
from iatreion.utils import logger

from .recorder import Recorder, TrainerReturn, get_display_name


@dataclass
class LastPredictions:
    y_true: NDArray
    y_score_list: list[NDArray]
    time: float


def get_last_predictions(named_recorders: dict[str, Recorder]) -> LastPredictions:
    time_list = []
    y_score_list = []
    for child in named_recorders.values():
        time_list.append(child.result.time[-1])
        y_score_list.append(child.result.y_all[-1].score)
    time = sum(time_list)
    y_true = child.result.y_all[-1].true
    return LastPredictions(y_true, y_score_list, time)


@dataclass
class FinalPredictions:
    y_true: NDArray
    y_pos_score_list: list[NDArray]
    weights: list[float]
    names: list[str]


def get_final_predictions(
    fold: int, named_recorders: dict[str, Recorder]
) -> FinalPredictions:
    names = []
    weights = []
    y_pos_score_list = []
    for name, child in named_recorders.items():
        names.append(name)
        finish = child.finish(calc_ci=False)
        weights.append(finish.final.f1)
        # HACK: Binary classification only
        y_pos_score_list.append(finish.final.y.score[:, 1])
        finish.log(f'{name}_inner_{fold}')
    y_true = finish.final.y.true
    return FinalPredictions(y_true, y_pos_score_list, weights, names)


def get_youden_threshold(y_true: NDArray, y_pos_score: NDArray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_pos_score)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def get_f1_threshold(y_true: NDArray, y_pos_score: NDArray) -> float:
    thresholds = np.arange(0.0, 1.02, 0.01)
    optimal_threshold = 0.5
    optimal_f1 = 0.0
    for threshold in thresholds:
        y_pred = (y_pos_score >= threshold).astype(int)
        # HACK: Binary classification only, set labels to [0, 1]
        current_f1 = f1_score(
            y_true, y_pred, labels=[0, 1], average='macro', zero_division=np.nan
        )
        if current_f1 > optimal_f1:
            optimal_f1 = current_f1
            optimal_threshold = threshold
    return optimal_threshold


def get_thresholds(y_true: NDArray, y_pos_score: NDArray) -> dict[str, float]:
    return {
        'youden_index': get_youden_threshold(y_true, y_pos_score),
        'f1_score': get_f1_threshold(y_true, y_pos_score),
    }


def get_meta_model(
    config: TrainConfig, fold: int, final: FinalPredictions
) -> LogisticRegression:
    meta_model = LogisticRegression(
        penalty='l2', C=0.5, random_state=42, solver='lbfgs'
    ).fit(np.c_[*final.y_pos_score_list], final.y_true)

    weights = meta_model.coef_[0]
    intercept = meta_model.intercept_[0]
    width = max(len(name) for name in final.names)
    with config.logging(f'weights_stacking_{fold}'):
        for idx, name in enumerate(final.names):
            logger.info(f'Weight for {f"{name}:":{width + 1}} {weights[idx]:.4f}')
        logger.info(f'Intercept (Bias): {intercept:.4f}')

    return meta_model


def aggregate_pos_scores(
    y_pos_score_list: list[NDArray],
    *,
    weights: list[float] | None = None,
    meta_model: LogisticRegression | None = None,
) -> NDArray:
    if meta_model is not None:
        y_pos_score = meta_model.predict_proba(np.c_[*y_pos_score_list])[:, 1]
    else:
        y_pos_score = np.average(y_pos_score_list, axis=0, weights=weights)
    return y_pos_score


def aggregate_scores(
    y_score_list: list[NDArray],
    *,
    weights: list[float] | None = None,
    meta_model: LogisticRegression | None = None,
) -> tuple[NDArray, list[float], float]:
    if meta_model is not None:
        # HACK: Binary classification only
        y_score = meta_model.predict_proba(
            np.hstack([score[:, [1]] for score in y_score_list])
        )
        norm_weights, bias = meta_model.coef_[0], meta_model.intercept_[0].item()
    else:
        y_score = np.average(y_score_list, axis=0, weights=weights)
        if weights is not None:
            norm_weights = np.array(weights) / sum(weights)
        else:
            n_total = len(y_score_list)
            norm_weights = np.full(n_total, 1 / n_total)
        bias = 0
    return y_score, norm_weights.tolist(), bias


def aggregate(
    fold: int,
    recorders: dict[str, Recorder],
    name: str,
    last: LastPredictions,
    final: FinalPredictions | None = None,
    *,
    weights: list[float] | None = None,
    meta_model: LogisticRegression | None = None,
) -> None:
    y_score, norm_weights, bias = aggregate_scores(
        last.y_score_list, weights=weights, meta_model=meta_model
    )
    thresholds: dict[str, float | None] = {'original': None}
    if final is not None:
        y_pos_score = aggregate_pos_scores(
            final.y_pos_score_list, weights=weights, meta_model=meta_model
        )
        thresholds |= get_thresholds(final.y_true, y_pos_score)
    for threshold_name, threshold in thresholds.items():
        recorder_name = f'{name}_{threshold_name}'
        recorder = recorders[recorder_name]
        recorder.record_weights_and_bias(norm_weights, bias)
        results = TrainerReturn(last.time, last.y_true, y_score, threshold=threshold)
        display_name = get_display_name(recorder_name)
        logger.info(
            f'[bold green]{display_name} (Fold {fold}):', extra={'markup': True}
        )
        logger.info(recorder.record(results))


def record_average(
    fold: int, recorders: dict[str, Recorder], outer_recorders: dict[str, Recorder]
) -> None:
    last = get_last_predictions(outer_recorders)
    aggregate(fold, recorders, 'all_simple_average', last)


def record_concats(
    fold: int,
    recorders: dict[str, Recorder],
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    last = get_last_predictions(outer_recorders)
    final = get_final_predictions(fold, inner_recorders)
    aggregate(fold, recorders, 'all_concats', last, final)


def record_stack(
    config: TrainConfig,
    fold: int,
    recorders: dict[str, Recorder],
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    last = get_last_predictions(outer_recorders)
    final = get_final_predictions(fold, inner_recorders)

    meta_model = get_meta_model(config, fold, final)
    aggregate(fold, recorders, 'all_simple_average', last, final)
    aggregate(
        fold, recorders, 'all_weighted_average', last, final, weights=final.weights
    )
    aggregate(fold, recorders, 'all_stacking', last, final, meta_model=meta_model)
