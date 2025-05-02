import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from collections import defaultdict

from iatreion.configs import DatasetConfig, ModelConfig, RrlConfig, TrainConfig
from iatreion.utils import logger

from .rrl.utils import read_csv, DBEncoder
from .rrl.models import RRL


def get_samples(dataset: DatasetConfig, model: ModelConfig, train: TrainConfig):
    data_path = dataset.prefix / f'{dataset.name}.data'
    info_path = dataset.prefix / f'{dataset.name}.info'
    X_df, y_df, f_df = read_csv(data_path, info_path, train.groups, train.label_pos, shuffle=True)

    db_enc = DBEncoder(f_df)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[model.ith_kfold]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    return db_enc, X_train, y_train, X_test, y_test


def get_data_loader(args: RrlConfig, pin_memory=False):
    db_enc, X_train, y_train, X_test, y_test = get_samples(args.dataset, args, args.train)

    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test))

    train_len = int(len(train_set) * 0.95)
    train_sub, valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if args.save_best:  # use validation set for model selections.
        train_set = train_sub

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader


def train_model(args: RrlConfig, advance=None):
    torch.manual_seed(42)

    writer = SummaryWriter(args.folder_path)


    db_enc, train_loader, valid_loader, _ = get_data_loader(args, pin_memory=True)

    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen

    rrl = RRL(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
              use_not=args.use_not,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              use_skip=args.skip,
              save_path=args.model,
              use_nlaf=args.nlaf,
              alpha=args.alpha,
              beta=args.beta,
              gamma=args.gamma,
              temperature=args.temp)

    rrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter,
        advance=advance)


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        use_not=saved_args['use_not'],
        estimated_grad=saved_args['estimated_grad'],
        use_skip=saved_args['use_skip'],
        use_nlaf=saved_args['use_nlaf'],
        alpha=saved_args['alpha'],
        beta=saved_args['beta'],
        gamma=saved_args['gamma'])
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


def test_model(args: RrlConfig):
    rrl = load_model(args.model)
    db_enc, train_loader, _, test_loader = get_data_loader(args)
    rrl.test(test_loader=test_loader, set_name='Test')
    if args.print_rule:
        with open(args.rrl_file, 'w') as rrl_file:
            rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std)
    else:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, mean=db_enc.mean, std=db_enc.std, display=False)
    
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
    logger.debug('\n\t{} of RRL  Model: {}'.format(metric, np.log(edge_cnt)))
