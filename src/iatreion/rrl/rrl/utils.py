import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold

from iatreion.configs import DatasetConfig, TrainConfig

pd.set_option('future.no_silent_downcasting', True)

def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().rsplit(maxsplit=1)
            f_list.append(tokens)
    return f_list


def get_group_mapping(groups):
    group_mapping = {}
    for group in groups:
        for name in group:
            group_mapping[name] = ''.join(group)
    return group_mapping


def read_csv(data_path, info_path, groups, label_pos, shuffle=False):
    f_list = read_info(info_path)
    names = [f[0] for f in f_list]
    D = pd.read_csv(data_path, names=names, dtype={'encrypted': str, 'Ab': str, 'A_type': str, 'A_type2': str})
    group_mapping = get_group_mapping(groups)
    if any(key[0] in 'LE' for key in group_mapping):
        D[label_pos] = D['A_type'].fillna(D[label_pos])
    elif any(key[0] == 'A' for key in group_mapping):
        D[label_pos] = D['A_type2'].fillna(D[label_pos])
    D[label_pos] = D[label_pos].map(group_mapping)
    D = D[D[label_pos].isin(list(group_mapping.values()))]
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_df = pd.DataFrame(f_list)
    y_df = D[label_pos]
    X_df = D.drop(['encrypted', 'Ab', 'A_type', 'A_type2'], axis=1)
    f_df = f_df.drop(f_df.index[[-4, -3, -2, -1]])
    return X_df, y_df, f_df


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, drop='first'):
        self.f_df = f_df
        self.label_enc = preprocessing.LabelEncoder()
        # TODO: drop = 'first' may be problematic for some datasets.
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = []
        self.y_fname = []
        self.discrete_flen = 0
        self.continuous_flen = 0
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        binary_data = X_df[self.f_df.loc[self.f_df[1] == 'binary', 0]]
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(float)
        return binary_data, discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        binary_data, discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df.values)
        self.y_fname = list(map(str, self.label_enc.classes_))

        if not binary_data.empty:
            self.X_fname += binary_data.columns.to_list()
            self.discrete_flen += len(binary_data.columns)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = list(
                self.feature_enc.get_feature_names_out(discrete_data.columns)
            )
            self.X_fname += feature_names
            self.discrete_flen += len(feature_names)
        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
            self.X_fname += continuous_data.columns.to_list()
            self.continuous_flen += continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        binary_data, discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values)

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = pd.DataFrame(self.feature_enc.transform(discrete_data).toarray())
        dfs = [binary_data, discrete_data, continuous_data]
        X_df = pd.concat([df for df in dfs if not df.empty], axis=1)
        return X_df.values, y


type Samples = tuple[DBEncoder, NDArray, NDArray, NDArray, NDArray]
type RawSamples = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


def get_samples(dataset: DatasetConfig, train: TrainConfig) -> Samples:
    data_path = dataset.data
    info_path = dataset.info
    X_df, y_df, f_df = read_csv(data_path, info_path, train.groups, train.label_pos, shuffle=True)

    db_enc = DBEncoder(f_df)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    kf = RepeatedStratifiedKFold(n_splits=train.n_splits, n_repeats=train.n_repeats, random_state=36851234)
    train_index, test_index = list(kf.split(X_df, y_df))[train.ith_kfold]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    return db_enc, X_train, y_train, X_test, y_test


def get_raw_samples(dataset: DatasetConfig, train: TrainConfig) -> RawSamples:
    data_path = dataset.data
    info_path = dataset.info
    X_df, y_df, _ = read_csv(data_path, info_path, train.groups, train.label_pos, shuffle=True)

    kf = RepeatedStratifiedKFold(n_splits=train.n_splits, n_repeats=train.n_repeats, random_state=36851234)
    train_index, test_index = list(kf.split(X_df, y_df))[train.ith_kfold]
    X_train = X_df.iloc[train_index]
    y_train = y_df.iloc[train_index]
    X_test = X_df.iloc[test_index]
    y_test = y_df.iloc[test_index]

    nunique = X_train.nunique(dropna=False)
    columns = nunique[nunique <= 1].index
    X_train = X_train.drop(columns=columns)
    X_test = X_test.drop(columns=columns)

    return X_train, y_train, X_test, y_test
