import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append([' '.join(tokens[:-1]), tokens[-1]])
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
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
            continuous_data = continuous_data.astype(np.float)
        return binary_data, discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        binary_data, discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df.values)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

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
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

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
