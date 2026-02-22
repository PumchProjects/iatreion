from collections.abc import Generator
from dataclasses import dataclass
from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
)
from numpy.typing import NDArray
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from iatreion.configs import DataName, DatasetConfig, TrainConfig
from iatreion.utils import encode_string, logger

pd.set_option('future.no_silent_downcasting', True)


def read_info(info_path) -> list[list[str]]:
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().rsplit(maxsplit=1)
            f_list.append(tokens)
    return f_list


def make_data_labels(
    D: pd.DataFrame,
    train: TrainConfig,
    group_columns: list[str],
    *,
    shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    base_pos = train.base_pos
    label_pos = train.label_pos

    D = D[~D.index.duplicated(keep=train.keep)]
    if shuffle:
        D = D.sample(frac=1, random_state=0)
    group_mapping = train.get_name_group_mapping()
    if base_pos:
        D.loc[:, label_pos] = D[base_pos].fillna(D[label_pos])
    D.loc[:, label_pos] = (
        D[label_pos].map(group_mapping, na_action='ignore').astype('string')
    )
    D = D[~D[label_pos].isna()]
    y_df = D[label_pos]
    X_df = D.drop(columns=group_columns)
    return X_df, y_df


def read_csv(
    name: DataName, dataset: DatasetConfig, train: TrainConfig, *, shuffle: bool = False
) -> tuple[pd.DataFrame, pd.Series, list[list[str]]]:
    data_path = dataset.get_data(name)
    info_path = dataset.get_info(name)

    f_list = read_info(info_path)
    group_columns = [name for name, type in f_list if type == 'label']

    names = [f[0] for f in f_list]
    dtype = {col: str for col in group_columns}
    D = pd.read_csv(data_path, names=names, index_col=0, dtype=dtype)
    X_df, y_df = make_data_labels(D, train, group_columns, shuffle=shuffle)
    f_list = f_list[1 : -len(group_columns)]

    X_df.rename(columns=encode_string, inplace=True)
    f_list = [[encode_string(name), type] for name, type in f_list]

    return X_df, y_df, f_list


def read_data(
    dataset: DatasetConfig, train: TrainConfig, *, shuffle: bool = False
) -> tuple[list[pd.DataFrame], list[pd.Series], pd.Series, list[pd.DataFrame]]:
    X_df, y_df, f_list = read_csv(dataset.names[0], dataset, train)
    ref_y_df = y_df
    X_dfs, y_dfs, f_dfs = [X_df], [y_df], [pd.DataFrame(f_list)]
    for name in dataset.names[1:]:
        X_df, y_df, f_list = read_csv(name, dataset, train)
        ref_y_df = ref_y_df[ref_y_df.index.intersection(y_df.index)]
        X_dfs.append(X_df)
        y_dfs.append(y_df)
        f_dfs.append(pd.DataFrame(f_list))
    if shuffle:
        ref_y_df = ref_y_df.sample(frac=1, random_state=0)
    return X_dfs, y_dfs, ref_y_df, f_dfs


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, train: TrainConfig, f_df: pd.DataFrame):
        self.train = train
        self.f_df = f_df
        self.label_enc = preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(
            categories='auto',
            drop='if_binary',
            sparse_output=False,
            handle_unknown='warn',
        )
        self.unordered_imp = SimpleImputer(strategy='most_frequent')
        self.ordered_imp = SimpleImputer(strategy='median')
        self.continuous_imp = SimpleImputer(strategy='mean')
        self.X_fname = []
        self.y_fname = []
        self.discrete_flen = 0
        self.continuous_flen = 0
        self.mean = None
        self.std = None

    def split_data_fine(self, X_df: pd.DataFrame):
        unordered_data = X_df[self.f_df.loc[self.f_df[1] == 'unordered', 0]]
        ordered_data = X_df[self.f_df.loc[self.f_df[1] == 'ordered', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(
                to_replace=r'.*\?.*', value=np.nan, regex=True
            )
            continuous_data = continuous_data.astype(float)
        return unordered_data, ordered_data, continuous_data

    def split_data_coarse(self, X_df: pd.DataFrame):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] != 'continuous', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        return discrete_data, continuous_data

    @staticmethod
    def concat_data(*data: pd.DataFrame) -> pd.DataFrame:
        dfs = [df for df in data if not df.empty]
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def try_resample(self, X, y, categorical):
        if self.train.over_sampler is None:
            return X, y
        if self.train.min_n_samples <= 0:
            strategy = 'auto'
        else:
            strategy = {
                cls: self.train.min_n_samples
                for cls in np.unique(y)
                if sum(y == cls) < self.train.min_n_samples
            }
        if not any(categorical):
            match self.train.over_sampler:
                case 'adasyn':
                    sm = ADASYN(sampling_strategy=strategy, random_state=42)
                case 'smote':
                    sm = SMOTE(sampling_strategy=strategy, random_state=42)
                case 'smotetomek':
                    sm = SMOTETomek(
                        sampling_strategy=strategy, random_state=42, n_jobs=4
                    )
                case 'smoteenn':
                    sm = SMOTEENN(sampling_strategy=strategy, random_state=42, n_jobs=4)
                case 'borderlinesmote-1':
                    sm = BorderlineSMOTE(
                        sampling_strategy=strategy, random_state=42, kind='borderline-1'
                    )
                case 'borderlinesmote-2':
                    sm = BorderlineSMOTE(
                        sampling_strategy=strategy, random_state=42, kind='borderline-2'
                    )
                case 'svmsmote':
                    sm = SVMSMOTE(sampling_strategy=strategy, random_state=42)
                case 'kmeanssmote':
                    sm = KMeansSMOTE(
                        sampling_strategy=strategy, random_state=42, n_jobs=4
                    )
        elif all(categorical):
            sm = SMOTEN(sampling_strategy=strategy, random_state=42)
        else:
            sm = SMOTENC(categorical, sampling_strategy=strategy, random_state=42)
        try:
            X, y = sm.fit_resample(X, y)
        except (ValueError, RuntimeError) as e:
            logger.warning(
                f'[bold yellow]Dataset might be too small, disabling SMOTE:[/] {e}',
                extra={'markup': True},
            )
        return X, y

    def fit_transform(self, X_df, y_df):
        # Encode string value to int index.
        y = self.label_enc.fit_transform(y_df)
        self.y_fname = list(map(str, self.label_enc.classes_))
        if not self.train.preprocess:
            self.X_fname = X_df.columns.to_list()
            return X_df.values, y

        unordered_data, ordered_data, continuous_data = self.split_data_fine(X_df)

        if not unordered_data.empty:
            # Use most frequent value as missing value for unordered columns.
            unordered_data = pd.DataFrame(
                self.unordered_imp.fit_transform(unordered_data),
                columns=unordered_data.columns,
            )
        if not ordered_data.empty:
            # Use median value as missing value for ordered columns.
            ordered_data = pd.DataFrame(
                self.ordered_imp.fit_transform(ordered_data),
                columns=ordered_data.columns,
            )
        if not continuous_data.empty:
            self.mean = continuous_data.mean()
            self.std = continuous_data.std()
            # Use mean as missing value for continuous columns if do not discretize them.
            continuous_data = pd.DataFrame(
                self.continuous_imp.fit_transform(continuous_data.values),
                columns=continuous_data.columns,
            )
            continuous_data = (continuous_data - self.mean) / self.std

        X = self.concat_data(unordered_data, ordered_data, continuous_data)
        categorical = [
            *(True for _ in range(X.shape[1] - self.continuous_flen)),
            *(False for _ in range(self.continuous_flen)),
        ]
        X, y = self.try_resample(X, y, categorical)

        discrete_data, continuous_data = self.split_data_coarse(X)
        data = []
        if not discrete_data.empty:
            # One-hot encoding
            data.append(self.feature_enc.fit_transform(discrete_data))
            feature_names = list(
                self.feature_enc.get_feature_names_out(discrete_data.columns)
            )
            self.X_fname = feature_names
            self.discrete_flen = len(feature_names)
        if not continuous_data.empty:
            data.append(continuous_data.values)
            self.X_fname += continuous_data.columns.to_list()
            self.continuous_flen = continuous_data.shape[1]

        return np.hstack(data), y

    def transform(self, X_df, y_df):
        if X_df is None or y_df is None:
            return None, None

        # Encode string value to int index.
        y = self.label_enc.transform(y_df)
        if not self.train.preprocess:
            return X_df.values, y

        unordered_data, ordered_data, continuous_data = self.split_data_fine(X_df)

        if not unordered_data.empty:
            # Use most frequent value as missing value for unordered columns.
            unordered_data = pd.DataFrame(
                self.unordered_imp.transform(unordered_data),
                columns=unordered_data.columns,
            )
        if not ordered_data.empty:
            # Use median value as missing value for ordered columns.
            ordered_data = pd.DataFrame(
                self.ordered_imp.transform(ordered_data),
                columns=ordered_data.columns,
            )

        data = []
        if not (discrete_data := self.concat_data(unordered_data, ordered_data)).empty:
            # One-hot encoding
            data.append(self.feature_enc.transform(discrete_data))
        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(
                self.continuous_imp.transform(continuous_data.values),
                columns=continuous_data.columns,
            )
            continuous_data = (continuous_data - self.mean) / self.std
            data.append(continuous_data.values)

        return np.hstack(data), y


@dataclass
class TrainStepContext:
    outer_fold: int
    inner_fold: int
    last: bool
    name: str

    db_enc: DBEncoder
    train_data: tuple[NDArray, NDArray]
    val_data: tuple[NDArray | None, NDArray | None]
    test_data: tuple[NDArray, NDArray]


type Samples = tuple[
    DBEncoder,
    NDArray,
    NDArray,
    NDArray | None,
    NDArray | None,
    NDArray,
    NDArray,
]
type RawSamples = tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame | None,
    pd.Series | None,
    pd.DataFrame,
    pd.Series,
]


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
    X_dfs, y_dfs, ref_y_df, f_dfs = read_data(dataset, train, shuffle=True)

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

    if train.aggregate == 'concat':
        X_dfs = [reduce(merge_X, X_dfs)]
        y_dfs = [reduce(merge_y, y_dfs)]
        f_dfs = [pd.concat(f_dfs, ignore_index=True)]

    outer_splitter = get_train_test(
        train.n_outer_splits, train.n_outer_repeats, ref_y_df
    )

    for outer_fold, (train_outer, test_outer) in enumerate(outer_splitter):
        if train.aggregate == 'stack':
            inner_splitter = chain(
                get_train_test(
                    train.n_inner_splits, train.n_inner_repeats, ref_y_df[train_outer]
                ),
                [(train_outer, test_outer)],
            )
        else:
            inner_splitter = [(train_outer, test_outer)]

        for inner_fold, (train_inner, test_inner) in enumerate(inner_splitter):
            last = (
                train.aggregate != 'stack'
                or inner_fold == train.n_inner_splits * train.n_inner_repeats
            )

            data_splitter = (
                ['all_concat'] if train.aggregate == 'concat' else dataset.names
            )

            for index, name in enumerate(data_splitter):
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

                db_enc = DBEncoder(train, f_df)
                yield TrainStepContext(
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    last=last,
                    name=name,
                    db_enc=db_enc,
                    train_data=db_enc.fit_transform(X_train, y_train),
                    val_data=db_enc.transform(X_val, y_val),
                    test_data=db_enc.transform(X_test, y_test),
                )


def get_raw_samples(
    dataset: DatasetConfig, train: TrainConfig
) -> Generator[RawSamples, None, None]:
    X_df, y_df, ref_y_df, f_df = read_data(dataset, train, shuffle=True)

    for train_arr, val_arr, test_arr in get_train_test(train, X_df, ref_y_df):
        X_train = X_df.iloc[train_arr]
        y_train = y_df.iloc[train_arr]
        X_val = None if val_arr is None else X_df.iloc[val_arr]
        y_val = None if val_arr is None else y_df.iloc[val_arr]
        X_test = X_df.iloc[test_arr]
        y_test = y_df.iloc[test_arr]

        yield X_train, y_train, X_val, y_val, X_test, y_test
