from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from iatreion.configs import RrlConfig
from iatreion.utils import logger

from .rrl.models import RRL
from .rrl.utils import TrainStepContext


def get_data_loader(args: RrlConfig, ctx: TrainStepContext, pin_memory=False):
    X_train, y_train = ctx.train_data
    X_val, y_val = ctx.val_data
    X_test, y_test = ctx.test_data

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train))
    valid_set = (
        None
        if X_val is None or y_val is None
        else TensorDataset(torch.tensor(X_val.astype(np.float32)), torch.tensor(y_val))
    )
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    valid_loader = (
        None
        if valid_set is None
        else DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    return ctx.db_enc, train_loader, valid_loader, test_loader


def load_model(save_model_callback: Callable[..., tuple[RRL, dict[str, Any], tuple[float, ...]]]) -> tuple[RRL, tuple[float, ...]]:
    rrl, state_dict, metrics = save_model_callback()
    rrl.net.load_state_dict(state_dict)
    return rrl, metrics


def train_model(args: RrlConfig, save_model_callback: Callable[..., tuple[RRL, dict[str, Any], tuple[float, ...]]], ctx: TrainStepContext):
    writer = SummaryWriter(args.folder_path)

    db_enc, train_loader, valid_loader, _ = get_data_loader(args, ctx, pin_memory=True)

    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen

    rrl = RRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              use_not=args.use_not,
              writer=writer,
              estimated_grad=args.estimated_grad,
              use_skip=args.skip,
              save_model_callback=save_model_callback,
              use_nlaf=args.nlaf,
              alpha=args.alpha,
              beta=args.beta,
              gamma=args.gamma,
              temperature=args.temp)
    
    y_true = ctx.train_data[1]
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_true),
        y=y_true
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float)

    rrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        log_iter=args.log_iter,
        save_interval=args.save_interval)
    
    if args.train.final and args.print_rule:
        rrl, metrics = load_model(save_model_callback)
        with open(args.train.log_dir / ctx.rrl_file, 'w', encoding='utf-8') as rrl_file:
            rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std, metrics=metrics)


def test_model(args: RrlConfig, save_model_callback: Callable[..., tuple[RRL, dict[str, Any], tuple[float, ...]]], ctx: TrainStepContext):
    rrl, metrics = load_model(save_model_callback)
    db_enc, train_loader, _, test_loader = get_data_loader(args, ctx)
    y_score, _, _ = rrl.test(test_loader=test_loader, set_name='Test')
    if args.print_rule:
        with open(args.train.log_dir / ctx.rrl_file, 'w', encoding='utf-8') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std, metrics=metrics)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, mean=db_enc.mean, std=db_enc.std, metrics=metrics, display=False)
    
    metric = 'Log(#Edges)'
    edge_cnt = 0
    connected_rid = defaultdict(lambda: set())
    ln = len(rrl.net.layer_list) - 1
    for rid, w in rule2weights:
        connected_rid[ln - abs(rid[0])].add(rid[1])
    while ln > 1:
        ln -= 1
        layer = rrl.net.layer_list[ln]
        for r in connected_rid[ln]:
            con_len = len(layer.rule_list[0])
            if r >= con_len:
                opt_id = 1
                r -= con_len
            else:
                opt_id = 0
            rule = layer.rule_list[opt_id][r]
            edge_cnt += len(rule)
            for rid in rule:
                connected_rid[ln - abs(rid[0])].add(rid[1])
    complexity = np.log(edge_cnt).item() if edge_cnt > 0 else np.nan
    logger.debug('\n\t{} of RRL  Model: {}'.format(metric, complexity))
    return y_score, complexity
