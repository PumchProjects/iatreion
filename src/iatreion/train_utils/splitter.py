from collections.abc import Generator, Iterable
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold, train_test_split

from iatreion.configs import DataName, DatasetConfig, ImportanceMethod, TrainConfig
from iatreion.configs.train import INNER_SPLIT_AGGREGATES, UNUSED_LABEL_NAME
from iatreion.exceptions import IatreionException
from iatreion.utils import encode_string

from .limix import LimiXWorkerClient, LimiXWorkerConfig
from .preprocessing import DBEncoder

pd.set_option('future.no_silent_downcasting', True)


def make_data_labels(
    D: pd.DataFrame, train: TrainConfig, group_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    label_name = train.label_name
    if label_name == UNUSED_LABEL_NAME:
        raise IatreionException('$label_name must be set', label_name='Label name')
    if label_name not in group_columns:
        raise IatreionException(
            'Label column "$label_name" is not marked as a label column.',
            label_name=label_name,
        )
    if label_name not in D.columns:
        raise IatreionException(
            'Label column "$label_name" not found in the data.',
            label_name=label_name,
        )
    D = D[~D.index.duplicated(keep=train.keep)]
    if train._shuffle:
        D = D.sample(frac=1, random_state=0)
    group_mapping = train.get_name_group_mapping()
    y_df = D[label_name].map(group_mapping, na_action='ignore')
    D = D.loc[~y_df.isna()]
    y_df = y_df.loc[D.index]
    X_df = D.drop(columns=group_columns)
    return X_df, y_df


def read_csv(
    name: DataName, dataset: DatasetConfig, train: TrainConfig
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data_path = dataset.get_data(name)
    info_path = dataset.get_info(name)

    f_df = pd.read_csv(info_path)
    group_columns = f_df.loc[f_df['type'] == 'label', 'name'].tolist()

    dtype = {col: str for col in group_columns}
    D = pd.read_csv(data_path, index_col=0, dtype=dtype)
    X_df, y_df = make_data_labels(D, train, group_columns)
    f_df = f_df.loc[f_df['type'] != 'label'].iloc[1:]

    if dataset._encode:
        X_df.rename(columns=encode_string, inplace=True)
        f_df['name'] = f_df['name'].map(encode_string)

    return X_df, y_df, f_df


def read_data(
    dataset: DatasetConfig, train: TrainConfig
) -> tuple[list[pd.DataFrame], list[pd.Series], pd.Series, list[pd.DataFrame]]:
    X_df, y_df, f_df = read_csv(dataset.names[0], dataset, train)
    ref_y_df = y_df
    X_dfs, y_dfs, f_dfs = [X_df], [y_df], [f_df]
    for name in dataset.names[1:]:
        X_df, y_df, f_df = read_csv(name, dataset, train)
        ref_y_df = ref_y_df.combine_first(y_df)
        X_dfs.append(X_df)
        y_dfs.append(y_df)
        f_dfs.append(f_df)
    if train._shuffle:
        ref_y_df = ref_y_df.sample(frac=1, random_state=0)
    return X_dfs, y_dfs, ref_y_df, f_dfs


def get_data_names(dataset: DatasetConfig, train: TrainConfig) -> list[str]:
    if train.aggregate in ('concat', 'calibrated-concat'):
        return ['all_concat']
    if not train.eval_names:
        return dataset.names

    invalid_names = sorted(set(train.eval_names) - set(dataset.names))
    if invalid_names:
        raise ValueError(
            'eval_names must be selected from dataset names. Unknown: '
            f'{", ".join(invalid_names)}.'
        )
    return train.eval_names


@dataclass
class FoldSpec:
    outer_fold: int
    inner_fold: int
    is_inner: bool
    train_index: pd.Index
    test_index: pd.Index


@dataclass
class TrainStepContext:
    outer_fold: int
    inner_fold: int
    is_inner: bool
    name: str
    train_index: pd.Index
    val_index: pd.Index | None
    test_index: pd.Index

    db_enc: DBEncoder
    train_data: tuple[NDArray, NDArray]
    val_data: tuple[NDArray | None, NDArray | None]
    test_data: tuple[NDArray, NDArray]
    test_mask: NDArray

    @property
    def rrl_file(self) -> str:
        if self.db_enc.train.final:
            return f'{self.name}.tsv'
        return f'rrl_{self.name}_{self.outer_fold}_{self.inner_fold}.tsv'

    def get_importance_file(self, method: ImportanceMethod) -> str:
        return f'score_{method}_{self.name}_{self.outer_fold}_{self.inner_fold}.json'

    @property
    def shap_file(self) -> str:
        return f'shap_{self.name}_{self.outer_fold}_{self.inner_fold}.npz'


def merge_data(
    X_dfs: list[pd.DataFrame], y_dfs: list[pd.Series], f_dfs: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    def merge_X(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return a.merge(b, how='outer', left_index=True, right_index=True)

    def merge_y(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.combine_first(b)

    X_df = reduce(merge_X, X_dfs)
    y_df = reduce(merge_y, y_dfs)
    f_df = pd.concat(f_dfs, ignore_index=True)
    return X_df, y_df, f_df


def get_train_test(
    n_splits: int, ref_y: pd.Series
) -> Generator[tuple[pd.Index, pd.Index], None, None]:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=36851234)
    for train, test in kf.split(ref_y, ref_y):
        yield ref_y.index[train], ref_y.index[test]


def get_cv_fold_specs(
    n_splits: int,
    ref_y: pd.Series,
    *,
    pool_index: pd.Index | None = None,
) -> list[FoldSpec]:
    y = ref_y if pool_index is None else ref_y.loc[pool_index]
    return [
        FoldSpec(
            outer_fold=fold,
            inner_fold=0,
            is_inner=False,
            train_index=train_index,
            test_index=test_index,
        )
        for fold, (train_index, test_index) in enumerate(get_train_test(n_splits, y))
    ]


def get_nested_fold_specs(train: TrainConfig, ref_y: pd.Series) -> list[FoldSpec]:
    if train.final:
        return [
            FoldSpec(
                outer_fold=0,
                inner_fold=0,
                is_inner=False,
                train_index=ref_y.index,
                test_index=pd.Index([]),
            )
        ]

    specs: list[FoldSpec] = []
    for outer_fold, (train_outer, test_outer) in enumerate(
        get_train_test(train.n_outer_splits, ref_y)
    ):
        if train.aggregate in INNER_SPLIT_AGGREGATES:
            for inner_fold, (train_inner, test_inner) in enumerate(
                get_train_test(train.n_inner_splits, ref_y.loc[train_outer])
            ):
                specs.append(
                    FoldSpec(
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        is_inner=True,
                        train_index=train_inner,
                        test_index=test_inner,
                    )
                )
            inner_fold = train.n_inner_folds
        else:
            inner_fold = 0

        specs.append(
            FoldSpec(
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                is_inner=False,
                train_index=train_outer,
                test_index=test_outer,
            )
        )
    return specs


def get_train_val(
    config: TrainConfig, y_df: pd.Series, train_index: pd.Index, X_index: pd.Index
) -> tuple[pd.Index, pd.Index | None]:
    train_index = train_index.intersection(X_index)
    val_index: pd.Index | None = None
    if config.val_size is not None:
        train_index, val_index = train_test_split(
            train_index,
            test_size=config.val_size,
            random_state=42,
            stratify=y_df[train_index],
        )
    return train_index, val_index


def get_train_iterator(
    dataset: DatasetConfig,
    train: TrainConfig,
    fold_specs: Iterable[FoldSpec] | None = None,
) -> Generator[TrainStepContext, None, None]:
    X_dfs, _, ref_y_df, f_dfs = read_data(dataset, train)
    data_names = get_data_names(dataset, train)
    limix_client = None
    if train.preprocess and train.missing_value_strategy == 'limix':
        limix_client = LimiXWorkerClient(
            LimiXWorkerConfig(
                python_path=train.limix_python_path,
                repo_path=train.limix_repo_path,
                model_path=train.limix_model_path,
                inference_config_path=train.resolved_limix_inference_config_path,
                device=train.limix_device,
                mask_prediction=True,
            ),
            name='LimiX imputer',
        )

    try:
        if train.aggregate in ('concat', 'calibrated-concat'):
            X_df, _, f_df = merge_data(X_dfs, [ref_y_df], f_dfs)
            data_frames = dict.fromkeys(data_names, (X_df, f_df))
        else:
            data_frames = dict(
                zip(dataset.names, zip(X_dfs, f_dfs, strict=True), strict=True)
            )

        specs = (
            list(fold_specs)
            if fold_specs is not None
            else get_nested_fold_specs(train, ref_y_df)
        )

        for spec in specs:
            for name in data_names:
                X_df, f_df = data_frames[name]
                train_final, val_final = get_train_val(
                    train, ref_y_df, spec.train_index, X_df.index
                )
                X_train = X_df.loc[train_final]
                y_train = ref_y_df.loc[train_final]
                X_val = None if val_final is None else X_df.loc[val_final]
                y_val = None if val_final is None else ref_y_df.loc[val_final]
                X_test = X_df.reindex(spec.test_index)
                y_test = ref_y_df.loc[spec.test_index]
                test_mask = np.isnan(X_test).all(axis=1)

                db_enc = DBEncoder(
                    train, f_df, cat_sep=dataset.cat_sep, limix_client=limix_client
                )
                train_data, val_data, test_data = db_enc.fit_transform(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                yield TrainStepContext(
                    outer_fold=spec.outer_fold,
                    inner_fold=spec.inner_fold,
                    is_inner=spec.is_inner,
                    name=name,
                    train_index=train_final,
                    val_index=val_final,
                    test_index=spec.test_index,
                    db_enc=db_enc,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    test_mask=test_mask,
                )
    finally:
        if limix_client is not None:
            limix_client.close()
