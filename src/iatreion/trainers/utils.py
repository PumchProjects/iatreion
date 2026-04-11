import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from iatreion.configs import TrainConfig
from iatreion.utils import logger

from .recorder import Recorder, TrainerReturn


def get_predictions(
    fold: int, inner_recorders: dict[str, Recorder]
) -> tuple[NDArray, list[NDArray], list[float], list[str]]:
    names = []
    weights = []
    y_pos_score_list = []
    for name, child in inner_recorders.items():
        names.append(name)
        finish = child.finish(calc_ci=False)
        weights.append(finish.final.f1)
        # HACK: Binary classification only
        y_pos_score_list.append(finish.final.y.score[:, 1])
        finish.log(f'{name}_inner_{fold}')
    y_true = finish.final.y.true
    return y_true, y_pos_score_list, weights, names


def get_threshold(y_true: NDArray, y_pos_score: NDArray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_pos_score)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def get_meta_model(
    config: TrainConfig,
    fold: int,
    y_true: NDArray,
    y_pos_score_list: list[NDArray],
    names: list[str],
) -> LogisticRegression:
    meta_model = LogisticRegression(
        penalty='l2', C=0.5, random_state=42, solver='lbfgs'
    ).fit(np.c_[*y_pos_score_list], y_true)

    weights = meta_model.coef_[0]
    intercept = meta_model.intercept_[0]
    width = max(len(name) for name in names)
    with config.logging(f'weights_stacking_{fold}'):
        for idx, name in enumerate(names):
            logger.info(f'Weight for {f"{name}:":{width + 1}} {weights[idx]:.4f}')
        logger.info(f'Intercept (Bias): {intercept:.4f}')

    return meta_model


def aggregate(
    recorder: Recorder,
    named_recorders: dict[str, Recorder],
    *,
    weights: list[float] | None = None,
    meta_model: LogisticRegression | None = None,
    threshold: float | None = None,
) -> str:
    time_list = []
    y_score_list = []
    for child in named_recorders.values():
        time_list.append(child.result.time[-1])
        y_score_list.append(child.result.y_all[-1].score)
    time = sum(time_list)
    y_true = child.result.y_all[-1].true
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
    recorder.record_weights_and_bias(norm_weights.tolist(), bias)
    return recorder.record(TrainerReturn(time, y_true, y_score, threshold=threshold))


def record_simple(
    fold: int, recorder: Recorder, outer_recorders: dict[str, Recorder]
) -> None:
    logger.info(f'[bold green]Simple Average (Fold {fold}):', extra={'markup': True})
    logger.info(aggregate(recorder, outer_recorders))


def record_all(
    fold: int,
    simple_recorder: Recorder,
    weighted_recorder: Recorder,
    stacking_recorder: Recorder,
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    y_true, y_pos_score_list, weights, names = get_predictions(fold, inner_recorders)

    y_pos_score = np.average(y_pos_score_list, axis=0)
    threshold = get_threshold(y_true, y_pos_score)
    logger.info(f'[bold green]Simple Average (Fold {fold}):', extra={'markup': True})
    logger.info(aggregate(simple_recorder, outer_recorders, threshold=threshold))

    y_pos_score = np.average(y_pos_score_list, axis=0, weights=weights)
    threshold = get_threshold(y_true, y_pos_score)
    logger.info(f'[bold green]Weighted Average (Fold {fold}):', extra={'markup': True})
    logger.info(
        aggregate(
            weighted_recorder, outer_recorders, weights=weights, threshold=threshold
        )
    )

    meta_model = get_meta_model(
        stacking_recorder.config, fold, y_true, y_pos_score_list, names
    )
    y_pos_score = meta_model.predict_proba(np.c_[*y_pos_score_list])[:, 1]
    threshold = get_threshold(y_true, y_pos_score)
    logger.info(f'[bold green]Stacking (Fold {fold}):', extra={'markup': True})
    logger.info(
        aggregate(
            stacking_recorder,
            outer_recorders,
            meta_model=meta_model,
            threshold=threshold,
        )
    )
