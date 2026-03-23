from dataclasses import dataclass

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

from iatreion.configs import TrainConfig
from iatreion.utils import logger

from .limix import LimiXWorkerClient

type EncodedData = tuple[NDArray, NDArray]
type OptionalEncodedData = tuple[NDArray | None, NDArray | None]


@dataclass(slots=True)
class _FrameSplits:
    train: pd.DataFrame
    val: pd.DataFrame | None
    test: pd.DataFrame


class DBEncoder:
    """Shared preprocessing and feature-encoding logic for training."""

    def __init__(
        self,
        train: TrainConfig,
        f_df: pd.DataFrame,
        *,
        cat_sep: str = ';',
        limix_client: LimiXWorkerClient | None = None,
    ) -> None:
        self.train = train
        self.f_df = f_df
        self.cat_sep = cat_sep
        self.limix_client = limix_client

        self.label_enc = preprocessing.LabelEncoder()
        self.X_compl_fname: dict[int, str] = {}
        self.X_fname: list[str] = []
        self.y_fname: list[str] = []
        self.binary_flen = 0
        self.categorical_flen = 0
        self.numeric_flen = 0
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self._continuous_mean = pd.Series(dtype=float)
        self._continuous_std = pd.Series(dtype=float)

        self.unordered_columns = self._get_columns('unordered')
        self.ordered_columns = self._get_columns('ordered')
        self.continuous_columns = self._get_columns('continuous')
        self.discrete_columns = [*self.unordered_columns, *self.ordered_columns]
        self.feature_columns = [*self.discrete_columns, *self.continuous_columns]
        self.category_labels = self._build_category_labels()
        self.binary_discrete_columns = [
            name for name in self.discrete_columns if self._category_count(name) <= 2
        ]
        self.categorical_discrete_columns = [
            name for name in self.discrete_columns if self._category_count(name) > 2
        ]

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> tuple[EncodedData, OptionalEncodedData, OptionalEncodedData]:
        y_train_encoded = self.label_enc.fit_transform(y_train)
        self.y_fname = list(map(str, self.label_enc.classes_))
        y_val_encoded = None if y_val is None else self.label_enc.transform(y_val)
        y_test_encoded = None if y_test is None else self.label_enc.transform(y_test)

        test_frame = (
            pd.DataFrame(index=pd.Index([], dtype=X_train.index.dtype))
            if X_test is None
            else X_test
        )

        if not self.train.preprocess:
            self.X_fname = X_train.columns.to_list()
            return (
                (X_train.values, y_train_encoded),
                self._pack_optional_output(X_val, y_val_encoded),
                self._pack_optional_output(test_frame, y_test_encoded),
            )

        frames = _FrameSplits(
            train=self._prepare_frame(X_train),
            val=None if X_val is None else self._prepare_frame(X_val),
            test=self._prepare_frame(test_frame),
        )

        frames = self._apply_missing_value_strategy(frames, y_train_encoded)
        frames = self._normalize_continuous_data(frames)
        frames.train, y_train_encoded = self._try_resample_train(
            frames.train, y_train_encoded
        )

        train_output = self._encode_output_frame(frames.train, fit=True)
        val_output = (
            None if frames.val is None else self._encode_output_frame(frames.val)
        )
        test_output = self._encode_output_frame(frames.test)
        return (
            (train_output, y_train_encoded),
            self._pack_optional_output_array(val_output, y_val_encoded),
            self._pack_optional_output_array(test_output, y_test_encoded),
        )

    def _get_columns(self, *types: str) -> list[str]:
        return self.f_df.loc[self.f_df['type'].isin(types), 'name'].tolist()

    def _build_category_labels(self) -> dict[str, tuple[str, ...]]:
        labels: dict[str, tuple[str, ...]] = {}
        cat_rows = self.f_df.loc[self.f_df['type'].isin({'unordered', 'ordered'})]
        for row in cat_rows.itertuples(index=False):
            raw = '' if pd.isna(row.categories) else str(row.categories)
            labels[row.name] = tuple(raw.split(self.cat_sep)) if raw else ()
        return labels

    def _category_count(self, name: str) -> int:
        if labels := self.category_labels.get(name):
            return len(labels)
        return 1

    def _category_label(self, name: str, code: int) -> str:
        labels = self.category_labels.get(name, ())
        if 0 <= code < len(labels):
            return labels[code]
        return str(code)

    def _prepare_frame(self, X_df: pd.DataFrame) -> pd.DataFrame:
        frame = X_df.loc[:, self.feature_columns].copy()
        return frame.apply(pd.to_numeric, errors='coerce').astype(float)

    def _apply_missing_value_strategy(
        self,
        frames: _FrameSplits,
        y_train: NDArray,
    ) -> _FrameSplits:
        match self.train.missing_value_strategy:
            case 'simple':
                return self._simple_impute(frames)
            case 'limix':
                return self._limix_impute(frames, y_train)
            case 'none':
                return frames

    def _simple_impute(self, frames: _FrameSplits) -> _FrameSplits:
        fill_values: dict[str, float] = {}
        for name in self.unordered_columns:
            mode = frames.train[name].dropna().mode()
            fill_values[name] = np.nan if mode.empty else float(mode.iloc[0])
        for name in self.ordered_columns:
            fill_values[name] = float(frames.train[name].median(skipna=True))
        for name in self.continuous_columns:
            fill_values[name] = float(frames.train[name].mean(skipna=True))

        for frame in (frames.train, frames.val, frames.test):
            if frame is None:
                continue
            frame.fillna(fill_values, inplace=True)
            self._snap_categorical_columns(frame, self.ordered_columns)
        return frames

    def _limix_impute(self, frames: _FrameSplits, y_train: NDArray) -> _FrameSplits:
        if self.limix_client is None:
            raise ValueError('LimiX imputation requires a configured LimiX worker.')

        combined_target, target_lengths = self._concat_target_frames(frames)
        combined = pd.concat([frames.train, combined_target], axis=0)
        if not combined.isna().to_numpy().any():
            return frames

        self.limix_client.mark_dirty()
        prediction = self.limix_client.predict(
            combined_target.to_numpy(dtype=np.float32),
            frames.train.to_numpy(dtype=np.float32),
            y_train,
            task_type='Regression',
        )
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise RuntimeError('LimiX imputer did not return reconstructed features.')

        reconstructed = np.asarray(prediction[1], dtype=float)
        if reconstructed.shape != combined.shape:
            raise RuntimeError(
                'Unexpected reconstructed feature shape from LimiX imputer: '
                f'expected {combined.shape}, got {reconstructed.shape}.'
            )

        filled = combined.to_numpy(dtype=float, copy=True)
        missing_mask = np.isnan(filled)
        filled[missing_mask] = reconstructed[missing_mask]
        combined_filled = pd.DataFrame(
            filled,
            index=combined.index,
            columns=combined.columns,
        )
        self._snap_categorical_columns(combined_filled, self.discrete_columns)

        frames.train = combined_filled.iloc[: len(frames.train)].copy()
        offset = len(frames.train)
        if frames.val is not None:
            size = target_lengths[0]
            frames.val = combined_filled.iloc[offset : offset + size].copy()
            offset += size
        frames.test = combined_filled.iloc[offset : offset + target_lengths[-1]].copy()
        return frames

    def _concat_target_frames(
        self, frames: _FrameSplits
    ) -> tuple[pd.DataFrame, tuple[int, int]]:
        val_frame = frames.val
        if val_frame is None:
            target = frames.test.copy()
            return target, (0, len(frames.test))
        target = pd.concat([val_frame, frames.test], axis=0)
        return target, (len(val_frame), len(frames.test))

    def _snap_categorical_columns(
        self, frame: pd.DataFrame, columns: list[str]
    ) -> None:
        for name in columns:
            if name not in frame.columns:
                continue
            values = frame[name].to_numpy(dtype=float, copy=True)
            valid_mask = ~np.isnan(values)
            if not valid_mask.any():
                continue
            upper = self._category_count(name) - 1
            snapped = np.floor(values[valid_mask] + 0.5)
            values[valid_mask] = np.clip(snapped, 0, upper)
            frame.loc[:, name] = values

    def _normalize_continuous_data(self, frames: _FrameSplits) -> _FrameSplits:
        if not self.continuous_columns:
            return frames

        if self.train.normalize_continuous:
            mean = frames.train[self.continuous_columns].mean()
            std = frames.train[self.continuous_columns].std().replace(0, 1.0)
            std = std.fillna(1.0) + 1e-8
            for frame in (frames.train, frames.val, frames.test):
                if frame is None:
                    continue
                frame.loc[:, self.continuous_columns] = (
                    frame[self.continuous_columns] - mean
                ) / std
            self._continuous_mean = mean.fillna(0.0)
            self._continuous_std = std
        else:
            index = pd.Index(self.continuous_columns)
            self._continuous_mean = pd.Series(0.0, index=index)
            self._continuous_std = pd.Series(1.0, index=index)
        return frames

    def _try_resample_train(
        self, X: pd.DataFrame, y: NDArray
    ) -> tuple[pd.DataFrame, NDArray]:
        if self.train.over_sampler is None:
            return X, y

        if self.train.min_n_samples <= 0:
            strategy: str | dict[int, int] = 'auto'
        else:
            strategy = {
                int(cls): self.train.min_n_samples
                for cls in np.unique(y)
                if np.sum(y == cls) < self.train.min_n_samples
            }
            if not strategy:
                return X, y

        categorical = [name in self.discrete_columns for name in X.columns]
        if not any(categorical):
            match self.train.over_sampler:
                case 'adasyn':
                    sampler = ADASYN(sampling_strategy=strategy, random_state=42)
                case 'smote':
                    sampler = SMOTE(sampling_strategy=strategy, random_state=42)
                case 'smotetomek':
                    sampler = SMOTETomek(
                        sampling_strategy=strategy, random_state=42, n_jobs=4
                    )
                case 'smoteenn':
                    sampler = SMOTEENN(
                        sampling_strategy=strategy, random_state=42, n_jobs=4
                    )
                case 'borderlinesmote-1':
                    sampler = BorderlineSMOTE(
                        sampling_strategy=strategy,
                        random_state=42,
                        kind='borderline-1',
                    )
                case 'borderlinesmote-2':
                    sampler = BorderlineSMOTE(
                        sampling_strategy=strategy,
                        random_state=42,
                        kind='borderline-2',
                    )
                case 'svmsmote':
                    sampler = SVMSMOTE(sampling_strategy=strategy, random_state=42)
                case 'kmeanssmote':
                    sampler = KMeansSMOTE(
                        sampling_strategy=strategy, random_state=42, n_jobs=4
                    )
        elif all(categorical):
            sampler = SMOTEN(sampling_strategy=strategy, random_state=42)
        else:
            sampler = SMOTENC(
                categorical_features=categorical,
                sampling_strategy=strategy,
                random_state=42,
            )

        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        except (ValueError, RuntimeError) as error:
            logger.warning(
                f'[bold yellow]Dataset might be too small, disabling SMOTE:[/] {error}',
                extra={'markup': True},
            )
            return X, y

        resampled = pd.DataFrame(X_resampled, columns=X.columns)
        return resampled, np.asarray(y_resampled)

    def _encode_output_frame(
        self, frame: pd.DataFrame, *, fit: bool = False
    ) -> NDArray:
        binary_parts: list[NDArray] = []
        categorical_parts: list[NDArray] = []
        numeric_parts: list[NDArray] = []

        compl_names: dict[int, str] = {}
        binary_names: list[str] = []
        categorical_names: list[str] = []
        numeric_names: list[str] = []

        inverse_mean = pd.Series(dtype=float)
        inverse_std = pd.Series(dtype=float)

        if self.train.discrete_processing == 'onehot':
            binary_array, binary_names, compl_names = self._one_hot_encode(frame)
            if binary_array.shape[1] > 0:
                binary_parts.append(binary_array)
        elif self.train.discrete_processing == 'none':
            binary_frame = frame.loc[:, self.binary_discrete_columns]
            categorical_frame = frame.loc[:, self.categorical_discrete_columns]
            binary_names = self.binary_discrete_columns
            categorical_names = self.categorical_discrete_columns
            if binary_frame.shape[1] > 0:
                binary_parts.append(binary_frame.to_numpy(dtype=float))
            if categorical_frame.shape[1] > 0:
                categorical_parts.append(categorical_frame.to_numpy(dtype=float))
                inverse_mean = pd.Series(0.0, index=categorical_names)
                inverse_std = pd.Series(1.0, index=categorical_names)
        else:
            binary_frame = frame.loc[:, self.binary_discrete_columns]
            binary_names = self.binary_discrete_columns
            if binary_frame.shape[1] > 0:
                binary_parts.append(binary_frame.to_numpy(dtype=float))
            scaled_numeric_frame, discrete_inverse_mean, discrete_inverse_std = (
                self._transform_numeric_discrete(frame)
            )
            if scaled_numeric_frame.shape[1] > 0:
                numeric_parts.append(scaled_numeric_frame.to_numpy(dtype=float))
                numeric_names.extend(scaled_numeric_frame.columns.to_list())
                inverse_mean = pd.concat([inverse_mean, discrete_inverse_mean])
                inverse_std = pd.concat([inverse_std, discrete_inverse_std])

        continuous_frame = frame.loc[:, self.continuous_columns]
        if continuous_frame.shape[1] > 0:
            numeric_parts.append(continuous_frame.to_numpy(dtype=float))
            numeric_names.extend(self.continuous_columns)
            inverse_mean = pd.concat(
                [inverse_mean, self._continuous_mean.reindex(self.continuous_columns)]
            )
            inverse_std = pd.concat(
                [inverse_std, self._continuous_std.reindex(self.continuous_columns)]
            )

        if fit:
            self.binary_flen = len(binary_names)
            self.categorical_flen = len(categorical_names)
            self.numeric_flen = len(numeric_names)
            self.X_fname = [*binary_names, *categorical_names, *numeric_names]
            self.X_compl_fname = compl_names
            self.mean = None if inverse_mean.empty else inverse_mean
            self.std = None if inverse_std.empty else inverse_std

        data_parts = [*binary_parts, *categorical_parts, *numeric_parts]
        if not data_parts:
            return np.empty((len(frame), 0))
        return np.hstack(data_parts)

    def _one_hot_name(self, name: str, code: int) -> str:
        return f'{name}_{code}_{self._category_label(name, code)}'

    def _one_hot_encode(
        self, frame: pd.DataFrame
    ) -> tuple[NDArray, list[str], dict[int, str]]:
        data_parts: list[NDArray] = []
        feature_names: list[str] = []
        compl_feature_names: dict[int, str] = {}

        for name in self.discrete_columns:
            series = frame[name]
            category_count = self._category_count(name)
            is_binary = category_count == 2
            codes = [1] if is_binary else list(range(category_count))
            for code in codes:
                values = np.where(
                    series.isna(),
                    np.nan,
                    (series.to_numpy(dtype=float) == float(code)).astype(float),
                )
                data_parts.append(values.reshape(-1, 1))
                if is_binary:
                    compl_feature_names[len(feature_names)] = self._one_hot_name(
                        name, 0
                    )
                feature_names.append(self._one_hot_name(name, code))

        if not data_parts:
            return np.empty((len(frame), 0)), [], {}
        return np.hstack(data_parts), feature_names, compl_feature_names

    def _transform_numeric_discrete(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        numeric = frame.loc[:, self.categorical_discrete_columns].copy()
        mean = pd.Series(dtype=float)
        std = pd.Series(dtype=float)

        for name in self.categorical_discrete_columns:
            category_range = max(self._category_count(name) - 1, 0)
            if category_range > 0:
                numeric.loc[:, name] = numeric[name] / category_range
            else:
                numeric.loc[:, name] = numeric[name].where(numeric[name].isna(), 0.0)
            mean.loc[name] = 0.0
            std.loc[name] = 1.0 if category_range == 0 else float(category_range)
        return numeric, mean, std

    @staticmethod
    def _pack_optional_output(
        X_df: pd.DataFrame | None, y: NDArray | None
    ) -> OptionalEncodedData:
        if X_df is None or y is None:
            return None, None
        return X_df.values, y

    @staticmethod
    def _pack_optional_output_array(
        X: NDArray | None, y: NDArray | None
    ) -> OptionalEncodedData:
        if X is None or y is None:
            return None, None
        return X, y
