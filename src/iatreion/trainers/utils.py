import numpy as np
from sklearn.linear_model import LogisticRegression

from iatreion.configs import TrainConfig
from iatreion.utils import logger

from .recorder import FinalRecord, Recorder


def get_meta_model(
    config: TrainConfig, fold: int, named_records: dict[str, FinalRecord]
) -> LogisticRegression:
    # HACK: Binary classification only
    width = 0
    y_score_list = []
    for name, record in named_records.items():
        width = max(width, len(name))
        y_score_list.append(record.y_score[:, [1]])
    y_true = record.y_true

    meta_model = LogisticRegression(
        penalty='l2', C=0.5, random_state=42, solver='lbfgs'
    ).fit(np.hstack(y_score_list), y_true)

    weights = meta_model.coef_[0]
    intercept = meta_model.intercept_[0]
    with config.logging(f'weights_stacking_{fold}'):
        for idx, name in enumerate(named_records.keys()):
            logger.info(f'Weight for {f"{name}:":{width + 1}} {weights[idx]:.4f}')
        logger.info(f'Intercept (Bias): {intercept:.4f}')

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
        y_score_list.append(child.result.y_score_all[-1])
    time = sum(time_list)
    y_true = child.result.y_true_all[-1]
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
    return recorder.record((time, y_true, y_score, {}))


def record_simple(
    fold: int, recorder: Recorder, outer_recorders: dict[str, Recorder]
) -> None:
    logger.info(f'[bold green]Simple Average (Fold {fold}):', extra={'markup': True})
    logger.info(aggregate(recorder, outer_recorders))


def record_weighted_and_stacking(
    fold: int,
    weighted_recorder: Recorder,
    stacking_recorder: Recorder,
    inner_recorders: dict[str, Recorder],
    outer_recorders: dict[str, Recorder],
) -> None:
    weights = []
    named_records = {}
    for name, child in inner_recorders.items():
        finish = child.finish(calc_ci=False)
        weights.append(finish.final.f1)
        named_records[name] = finish.final
        finish.log(f'{name}_inner_{fold}')
    logger.info(f'[bold green]Weighted Average (Fold {fold}):', extra={'markup': True})
    logger.info(aggregate(weighted_recorder, outer_recorders, weights=weights))

    meta_model = get_meta_model(stacking_recorder.config, fold, named_records)
    logger.info(f'[bold green]Stacking (Fold {fold}):', extra={'markup': True})
    logger.info(aggregate(stacking_recorder, outer_recorders, meta_model=meta_model))
