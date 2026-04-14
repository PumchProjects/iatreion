from collections.abc import Generator
from dataclasses import dataclass
from functools import reduce
from itertools import chain

import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from iatreion.configs import DataName, DatasetConfig, ImportanceMethod, TrainConfig
from iatreion.utils import encode_string

from .limix import LimiXWorkerClient, LimiXWorkerConfig
from .preprocessing import DBEncoder

pd.set_option('future.no_silent_downcasting', True)


def make_data_labels(
    D: pd.DataFrame, train: TrainConfig, group_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    base_pos = train._base_pos
    label_pos = train._label_pos

    D = D[~D.index.duplicated(keep=train.keep)]
    if train._shuffle:
        D = D.sample(frac=1, random_state=0)
    group_mapping = train.get_name_group_mapping()
    if base_pos:
        D.loc[:, label_pos] = D[base_pos].fillna(D[label_pos])
    D.loc[:, label_pos] = D[label_pos].map(group_mapping, na_action='ignore')
    D = D[~D[label_pos].isna()]
    y_df = D[label_pos]
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
    f_df = f_df.iloc[1 : -len(group_columns)]

    if train._encode:
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
        ref_y_df = ref_y_df[ref_y_df.index.intersection(y_df.index)]
        X_dfs.append(X_df)
        y_dfs.append(y_df)
        f_dfs.append(f_df)
    if train._shuffle:
        ref_y_df = ref_y_df.sample(frac=1, random_state=0)
    return X_dfs, y_dfs, ref_y_df, f_dfs


def get_data_names(dataset: DatasetConfig, train: TrainConfig) -> list[str]:
    if train.aggregate in ('concat', 'concats'):
        # HACK: Delicate config should change the name according to train.final
        return [dataset.name_str] if train.final else ['all_concat']
    else:
        return dataset.names


@dataclass
class TrainStepContext:
    outer_fold: int
    inner_fold: int
    is_inner: bool
    name: str

    db_enc: DBEncoder
    train_data: tuple[NDArray, NDArray]
    val_data: tuple[NDArray | None, NDArray | None]
    test_data: tuple[NDArray, NDArray]

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
        merged = pd.merge(
            a.to_frame('left'),
            b.to_frame('right'),
            how='outer',
            left_index=True,
            right_index=True,
        )
        return merged['left'].combine_first(merged['right'])

    X_df = reduce(merge_X, X_dfs)
    y_df = reduce(merge_y, y_dfs)
    f_df = pd.concat(f_dfs, ignore_index=True)
    return X_df, y_df, f_df


def get_train_test(
    n_splits: int, n_repeats: int, ref_y: pd.Series
) -> Generator[tuple[pd.Index, pd.Index], None, None]:
    kf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=36851234,
    )
    for train, test in kf.split(ref_y, ref_y):
        yield ref_y.index[train], ref_y.index[test]


def get_train_val(
    config: TrainConfig, y_df: pd.Series, train_index: pd.Index, test_index: pd.Index
) -> tuple[pd.Index, pd.Index | None]:
    if not config.true_ref:
        train_index = y_df.index.difference(test_index)
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
    dataset: DatasetConfig, train: TrainConfig
) -> Generator[TrainStepContext, None, None]:
    X_dfs, y_dfs, ref_y_df, f_dfs = read_data(dataset, train)
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
        if train.aggregate in ('concat', 'concats'):
            X_df, y_df, f_df = merge_data(X_dfs, y_dfs, f_dfs)
            X_dfs, y_dfs, f_dfs = [X_df], [y_df], [f_df]

        if train.final:
            outer_splitter = [(ref_y_df.index, pd.Index([]))]
        else:
            outer_splitter = get_train_test(
                train.n_outer_splits, train.n_outer_repeats, ref_y_df
            )

        for outer_fold, (train_outer, test_outer) in enumerate(outer_splitter):
            if train.aggregate in ('concats', 'stack') and not train.final:
                inner_splitter = chain(
                    get_train_test(
                        train.n_inner_splits,
                        train.n_inner_repeats,
                        ref_y_df[train_outer],
                    ),
                    [(train_outer, test_outer)],
                )
            else:
                inner_splitter = [(train_outer, test_outer)]

            for inner_fold, (train_inner, test_inner) in enumerate(inner_splitter):
                is_inner = (
                    train.aggregate in ('concats', 'stack')
                    and inner_fold < train.n_inner_folds
                )

                for index, name in enumerate(data_names):
                    X_df, y_df, f_df = X_dfs[index], y_dfs[index], f_dfs[index]
                    test_union = test_inner.union(test_outer)
                    train_final, val_final = get_train_val(
                        train, y_df, train_inner, test_union
                    )
                    X_train = X_df.loc[train_final]
                    y_train = y_df.loc[train_final]
                    X_val = None if val_final is None else X_df.loc[val_final]
                    y_val = None if val_final is None else y_df.loc[val_final]
                    X_test = X_df.loc[test_inner]
                    y_test = y_df.loc[test_inner]

                    db_enc = DBEncoder(
                        train, f_df, cat_sep=dataset.cat_sep, limix_client=limix_client
                    )
                    train_data, val_data, test_data = db_enc.fit_transform(
                        X_train, y_train, X_val, y_val, X_test, y_test
                    )
                    yield TrainStepContext(
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        is_inner=is_inner,
                        name=name,
                        db_enc=db_enc,
                        train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                    )
    finally:
        if limix_client is not None:
            limix_client.close()
