import numpy as np
from sklearn.linear_model import LogisticRegression

from iatreion.utils import logger

from .recorder import Recorder


def get_meta_model(
    fold: int, named_recorders: dict[str, Recorder]
) -> LogisticRegression:
    # HACK: Binary classification only
    width = 0
    y_score_list = []
    for name, recorder in named_recorders.items():
        width = max(width, len(name))
        y_score_list.append(np.vstack(recorder.y_score_all)[:, [1]])
    y_true = np.concatenate(recorder.y_true_all)

    meta_model = LogisticRegression(
        C=0.5, l1_ratio=0, random_state=42, solver='lbfgs'
    ).fit(np.hstack(y_score_list), y_true)

    weights = meta_model.coef_[0]
    intercept = meta_model.intercept_[0]
    with recorder.config.logging(f'stacking_{fold}'):
        logger.info(f'Intercept (Bias): {intercept:.4f}')
        for idx, name in enumerate(named_recorders.keys()):
            logger.info(f'Weight for {f"{name}:":{width + 1}} {weights[idx]:.4f}')

    return meta_model


def aggregate(
    recorder: Recorder,
    named_recorders: dict[str, Recorder],
    *,
    weights: list[float] | None = None,
    meta_model: LogisticRegression | None = None,
) -> str:
    time_list = []
    y_score_list = []
    for child in named_recorders.values():
        time_list.append(child.result.time[-1])
        y_score_list.append(child.y_score_all[-1])
    time = sum(time_list)
    y_true = child.y_true_all[-1]
    if meta_model is not None:
        # HACK: Binary classification only
        y_score = meta_model.predict_proba(
            np.hstack([score[:, [1]] for score in y_score_list])
        )
    else:
        y_score = np.average(y_score_list, axis=0, weights=weights)
    return recorder.record((time, y_true, y_score, {}))


def record_simple(recorder: Recorder, outer_recorders: dict[str, Recorder]) -> None:
    logger.info(aggregate(recorder, outer_recorders))


def record_weighted(
    fold: int,
    recorder: Recorder,
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    weights = []
    for name, child in inner_recorders.items():
        finish = child.finish()
        weights.append(finish.final.f1)
        finish.log(f'{name}_train_{fold}')
    logger.info(aggregate(recorder, outer_recorders, weights=weights))


def record_stacking(
    fold: int,
    recorder: Recorder,
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    meta_model = get_meta_model(fold, inner_recorders)
    logger.info(aggregate(recorder, outer_recorders, meta_model=meta_model))
