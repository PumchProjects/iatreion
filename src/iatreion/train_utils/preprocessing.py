from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from iatreion.configs import TrainConfig

from .feature_selection import FeatureSelectionArtifact, SupervisedFeatureSelector
from .imputation import SimpleImputerArtifact, SimpleImputerColumn
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

        self.X_compl_fname: dict[int, str] = {}
        self.X_fname: list[str] = []
        self.y_fname: list[str] = []
        self.binary_flen = 0
        self.categorical_flen = 0
        self.numeric_flen = 0
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self.simple_imputer: SimpleImputerArtifact | None = None
        self.feature_selection: FeatureSelectionArtifact | None = None
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
        self.y_fname = self.train.group_labels
        y_train_encoded = self._encode_labels(y_train)
        y_val_encoded = None if y_val is None else self._encode_labels(y_val)
        y_test_encoded = None if y_test is None else self._encode_labels(y_test)

        test_frame = X_train.iloc[:0].copy() if X_test is None else X_test

        if not self.train.preprocess:
            X_train, y_train_encoded = self._try_undersample_train(
                X_train, y_train_encoded
            )
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

        frames = self._apply_feature_selection(frames, y_train_encoded)
        frames = self._apply_missing_value_strategy(frames, y_train_encoded)
        frames = self._normalize_continuous_data(frames)
        frames.train, y_train_encoded = self._try_undersample_train(
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

    def _encode_labels(self, y: pd.Series) -> NDArray:
        return y.map(self.train.get_group_index_mapping()).to_numpy(dtype=int)

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

    def _apply_feature_selection(
        self,
        frames: _FrameSplits,
        y_train: NDArray,
    ) -> _FrameSplits:
        selector = SupervisedFeatureSelector(
            self.train,
            feature_columns=self.feature_columns,
            unordered_columns=self.unordered_columns,
            ordered_columns=self.ordered_columns,
            continuous_columns=self.continuous_columns,
            category_counts={
                name: self._category_count(name) for name in self.unordered_columns
            },
        )
        selector.fit(frames.train, y_train)
        selected = selector.selected_features
        self.feature_selection = selector.artifact
        self._restrict_feature_columns(selected)
        return _FrameSplits(
            train=selector.transform(frames.train),
            val=None if frames.val is None else selector.transform(frames.val),
            test=selector.transform(frames.test),
        )

    def _restrict_feature_columns(self, selected: list[str]) -> None:
        selected_set = set(selected)
        self.unordered_columns = [
            name for name in self.unordered_columns if name in selected_set
        ]
        self.ordered_columns = [
            name for name in self.ordered_columns if name in selected_set
        ]
        self.continuous_columns = [
            name for name in self.continuous_columns if name in selected_set
        ]
        self.discrete_columns = [*self.unordered_columns, *self.ordered_columns]
        self.feature_columns = [*self.discrete_columns, *self.continuous_columns]
        self.category_labels = {
            name: labels
            for name, labels in self.category_labels.items()
            if name in selected_set
        }
        self.binary_discrete_columns = [
            name for name in self.discrete_columns if self._category_count(name) <= 2
        ]
        self.categorical_discrete_columns = [
            name for name in self.discrete_columns if self._category_count(name) > 2
        ]

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
        columns: list[SimpleImputerColumn] = []
        for name in self.unordered_columns:
            mode = frames.train[name].dropna().mode()
            fill_value = np.nan if mode.empty else float(mode.iloc[0])
            columns.append(SimpleImputerColumn(name, fill_value))
        for name in self.ordered_columns:
            fill_value = float(frames.train[name].median(skipna=True))
            columns.append(
                SimpleImputerColumn(
                    name,
                    fill_value,
                    snap_upper=self._category_count(name) - 1,
                )
            )
        for name in self.continuous_columns:
            fill_value = float(frames.train[name].mean(skipna=True))
            columns.append(SimpleImputerColumn(name, fill_value))

        self.simple_imputer = SimpleImputerArtifact(columns)
        for frame in (frames.train, frames.val, frames.test):
            if frame is None:
                continue
            frame.loc[:, :] = self.simple_imputer.apply(
                frame,
                preserve_all_missing=False,
            )
        return frames

    def save_simple_imputer(self, path: Path) -> None:
        if self.simple_imputer is not None:
            self.simple_imputer.save(path)

    def save_feature_selection(self, path: Path) -> None:
        if self.feature_selection is not None:
            self.feature_selection.save(path)

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

    def _try_undersample_train(
        self, X: pd.DataFrame, y: NDArray
    ) -> tuple[pd.DataFrame, NDArray]:
        if self.train.under_sampler is None:
            return X, y

        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        target = self.train.target_n_samples or int(counts.min())
        selected_parts: list[NDArray[np.integer]] = []
        rng = np.random.default_rng(self.train.seed)

        for cls, count in zip(classes, counts, strict=True):
            cls_indices = np.flatnonzero(y == cls)
            if count > target:
                cls_indices = rng.choice(cls_indices, size=target, replace=False)
            selected_parts.append(cls_indices)

        selected = np.sort(np.concatenate(selected_parts))
        if selected.size == len(y):
            return X, y
        return X.iloc[selected].copy(), y[selected]

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
