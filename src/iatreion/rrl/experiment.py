from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from iatreion.configs import RrlConfig
from iatreion.train_utils import TrainStepContext
from iatreion.utils import logger, task

from .rrl.models import RRL


def _prepare_model_input(
    args: RrlConfig, X: NDArray
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    values = np.asarray(X, dtype=np.float32)
    if args.missing_aware_mode == 'improved':
        missing = np.isnan(values)
        mask = (~missing).astype(np.float32)
        values = np.where(missing, 0.0, values)
        return values, mask
    return values, np.ones_like(values, dtype=np.float32)


def get_data_loader(
    args: RrlConfig,
    X: NDArray,
    y: NDArray | None = None,
    *,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    X_value, X_mask = _prepare_model_input(args, X)
    if y is not None:
        dataset = TensorDataset(
            torch.tensor(X_value),
            torch.tensor(X_mask),
            torch.tensor(y),
        )
    else:
        dataset = TensorDataset(torch.tensor(X_value), torch.tensor(X_mask))
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory
    )


def train_model(
    args: RrlConfig,
    save_model_callback: Callable[..., None],
    ctx: TrainStepContext,
):
    args.folder_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.folder_path))

    train_loader = get_data_loader(args, *ctx.train_data, shuffle=True, pin_memory=True)
    if ctx.val_data[0] is not None and ctx.val_data[1] is not None:
        valid_loader = get_data_loader(args, *ctx.val_data, pin_memory=True)
    else:
        valid_loader = None

    db_enc = ctx.db_enc
    y_fname = db_enc.y_fname
    binary_flen = db_enc.binary_flen
    categorical_flen = db_enc.categorical_flen
    numeric_flen = db_enc.numeric_flen

    rrl = RRL(
        dim_list=[
            (binary_flen, categorical_flen + numeric_flen),
            *list(map(int, args.structure.split('@'))),
            len(y_fname),
        ],
        use_not=args.use_not,
        writer=writer,
        estimated_grad=args.estimated_grad,
        use_skip=args.skip,
        save_model_callback=save_model_callback,
        use_nlaf=args.nlaf,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temp,
        use_disjunction=not args.conjunction_only,
        use_missing_aware=args.missing_aware_mode == 'improved',
        coverage_tau=args.coverage_tau,
        coverage_kappa=args.coverage_kappa,
    )

    y_true = ctx.train_data[1]
    class_weights = None
    if args.weighted:
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_true),
            y=y_true,
        )
        class_weights = torch.tensor(class_weights_array, dtype=torch.float)
        logger.info(f'Using class weights: {class_weights.tolist()}')

    with task('Epoch:', args.epoch) as epoch_advance:
        rrl.train_model(
            epoch_advance=epoch_advance,
            data_loader=train_loader,
            valid_loader=valid_loader,
            lr=args.learning_rate,
            epoch=args.epoch,
            lr_decay_rate=args.lr_decay_rate,
            lr_decay_epoch=args.lr_decay_epoch,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            log_iter=args.log_iter,
            save_interval=args.save_interval,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
        )


def print_rules(
    args: RrlConfig,
    ctx: TrainStepContext,
    rrl: RRL,
    metrics: tuple[float, ...],
) -> Any:
    train_loader = get_data_loader(args, *ctx.train_data)
    db_enc = ctx.db_enc
    if args.print_rule:
        with open(
            args.train._log_dir / ctx.rrl_file, 'w', encoding='utf-8'
        ) as rrl_file:
            rule2weights = rrl.rule_print(
                db_enc.X_fname,
                db_enc.X_compl_fname,
                db_enc.y_fname,
                train_loader,
                file=rrl_file,
                mean=db_enc.mean,
                std=db_enc.std,
                metrics=metrics,
            )
    else:
        rule2weights = rrl.rule_print(
            db_enc.X_fname,
            db_enc.X_compl_fname,
            db_enc.y_fname,
            train_loader,
            mean=db_enc.mean,
            std=db_enc.std,
            metrics=metrics,
            display=False,
        )
    return rule2weights


def test_model(args: RrlConfig, X: NDArray, rrl: RRL):
    test_loader = get_data_loader(args, X)
    y_score = rrl.predict_proba(test_loader)
    return y_score


def calc_complexity(rrl: RRL, rule2weights: Any) -> float:
    metric = 'Log(#Edges)'
    edge_cnt = 0
    connected_rid = defaultdict(lambda: set())
    ln = len(rrl.net.layer_list) - 1
    for rid, _w in rule2weights:
        connected_rid[ln - abs(rid[0])].add(rid[1])
    while ln > 1:
        ln -= 1
        layer = rrl.net.layer_list[ln]
        for r in connected_rid[ln]:
            rule = layer.get_rule(r)
            edge_cnt += len(rule)
            for rid in rule:
                connected_rid[ln - abs(rid[0])].add(rid[1])
    complexity = np.log(edge_cnt).item() if edge_cnt > 0 else np.nan
    logger.debug(f'\n\t{metric} of RRL  Model: {complexity}')
    return complexity
