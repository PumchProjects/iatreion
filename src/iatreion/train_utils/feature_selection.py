from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from iatreion.utils import save_dict

if TYPE_CHECKING:
    from iatreion.configs import TrainConfig

FEATURE_SELECTION_SUFFIX = '.feature-selection.toml'


def get_feature_selection_path(rrl_path: Path) -> Path:
    return rrl_path.with_name(f'{rrl_path.stem}{FEATURE_SELECTION_SUFFIX}')


@dataclass(frozen=True)
class FeatureSelectionArtifact:
    method: str
    selected_features: list[str]
    dropped_features: list[str]
    ranked_features: list[str]
    scores: dict[str, float]
    keep_count: int
    keep_fraction: float
    score_aggregate: str
    params: dict[str, object]
    version: int = 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict(
            {
                'version': self.version,
                'method': self.method,
                'params': self.params,
                'keep_count': self.keep_count,
                'keep_fraction': self.keep_fraction,
                'score_aggregate': self.score_aggregate,
                'selected_features': self.selected_features,
                'dropped_features': self.dropped_features,
                'ranked_features': self.ranked_features,
                'scores': self.scores,
            },
            path,
        )


class SupervisedFeatureSelector:
    def __init__(
        self,
        train: TrainConfig,
        *,
        feature_columns: list[str],
        unordered_columns: list[str],
        ordered_columns: list[str],
        continuous_columns: list[str],
        category_counts: dict[str, int],
    ) -> None:
        self.train = train
        self.config = train.feature_selection
        self.feature_columns = feature_columns
        self.unordered_columns = set(unordered_columns)
        self.ordered_columns = set(ordered_columns)
        self.continuous_columns = set(continuous_columns)
        self.category_counts = category_counts
        self.selected_features = list(feature_columns)
        self.artifact: FeatureSelectionArtifact | None = None

    def fit(self, frame: pd.DataFrame, y: NDArray) -> None:
        method = self.config.method
        if method == 'none':
            self.artifact = None
            self.selected_features = list(self.feature_columns)
            return

        score_view, raw_features, discrete_features = self._make_score_view(frame)
        scores = self._score(score_view, y, discrete_features)
        feature_scores = self._aggregate_scores(raw_features, scores)
        ranked_features = self._rank_features(feature_scores)
        keep_count = self._keep_count(len(ranked_features))
        selected = set(ranked_features[:keep_count])

        self.selected_features = [
            name for name in self.feature_columns if name in selected
        ]
        self.artifact = FeatureSelectionArtifact(
            method=method,
            selected_features=self.selected_features,
            dropped_features=[
                name for name in self.feature_columns if name not in selected
            ],
            ranked_features=ranked_features,
            scores=feature_scores,
            keep_count=len(self.selected_features),
            keep_fraction=(
                len(self.selected_features) / len(self.feature_columns)
                if self.feature_columns
                else 0.0
            ),
            score_aggregate=self.config.score_aggregate,
            params=self._artifact_params(),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, self.selected_features].copy()

    def _keep_count(self, total: int) -> int:
        if total == 0:
            return 0
        if self.config.top_k is not None:
            keep = self.config.top_k
        else:
            keep = ceil(total * self.config.fraction)
        keep = max(keep, self.config.min_features)
        if self.config.max_features is not None:
            keep = min(keep, self.config.max_features)
        return min(keep, total)

    def _score(
        self, score_view: pd.DataFrame, y: NDArray, discrete_features: list[bool]
    ) -> NDArray:
        if score_view.shape[1] == 0:
            return np.empty(0, dtype=float)

        X = score_view.to_numpy(dtype=float)
        match self.config.method:
            case 'f_classif':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scores, _ = f_classif(X, y)
            case 'mutual_info':
                scores = mutual_info_classif(
                    X,
                    y,
                    discrete_features=np.asarray(discrete_features, dtype=bool),
                    random_state=self.train.seed,
                )
            case 'l1_logistic':
                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=float(self.config.c),
                    max_iter=5000,
                    random_state=self.train.seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(X, y)
                scores = np.abs(model.coef_).max(axis=0)
            case method:
                raise ValueError(f'Unknown feature-selection method: {method}.')
        return np.nan_to_num(
            np.asarray(scores, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _aggregate_scores(
        self, raw_features: list[str], scores: NDArray
    ) -> dict[str, float]:
        result = {name: 0.0 for name in self.feature_columns}
        if len(raw_features) == 0:
            return result

        score_frame = pd.DataFrame({'feature': raw_features, 'score': scores})
        grouped = score_frame.groupby('feature', sort=False)['score']
        if self.config.score_aggregate == 'mean':
            aggregated = grouped.mean()
        else:
            aggregated = grouped.max()

        for name, score in aggregated.items():
            result[name] = float(score)
        return result

    def _rank_features(self, scores: dict[str, float]) -> list[str]:
        order = {name: index for index, name in enumerate(self.feature_columns)}
        return sorted(
            self.feature_columns,
            key=lambda name: (-scores[name], order[name]),
        )

    def _make_score_view(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], list[bool]]:
        columns: dict[str, NDArray] = {}
        raw_features: list[str] = []
        discrete_features: list[bool] = []

        for name in self.feature_columns:
            series = frame[name]
            if name in self.unordered_columns:
                self._append_unordered(columns, raw_features, discrete_features, series)
            else:
                columns[name] = self._numeric_score_values(series)
                raw_features.append(name)
                discrete_features.append(name in self.ordered_columns)

        return pd.DataFrame(columns, index=frame.index), raw_features, discrete_features

    def _append_unordered(
        self,
        columns: dict[str, NDArray],
        raw_features: list[str],
        discrete_features: list[bool],
        series: pd.Series,
    ) -> None:
        name = str(series.name)
        values = series.fillna(self._mode_fill(series)).to_numpy(dtype=float)
        category_count = self.category_counts[name]
        codes = [1] if category_count == 2 else list(range(category_count))
        for code in codes:
            columns[f'{name}__{code}'] = (values == float(code)).astype(float)
            raw_features.append(name)
            discrete_features.append(True)

    def _numeric_score_values(self, series: pd.Series) -> NDArray:
        if series.name in self.continuous_columns:
            fill_value = series.mean(skipna=True)
        else:
            fill_value = series.median(skipna=True)
        if pd.isna(fill_value):
            fill_value = 0.0

        values = series.fillna(float(fill_value)).to_numpy(dtype=float)
        if series.name in self.ordered_columns and self.config.method != 'l1_logistic':
            return values

        mean = np.mean(values)
        std = np.std(values)
        if not np.isfinite(std) or std == 0:
            return np.zeros_like(values, dtype=float)
        return (values - mean) / std

    def _artifact_params(self) -> dict[str, object]:
        if self.config.method != 'l1_logistic':
            return {}
        return {'c': float(self.config.c)}

    @staticmethod
    def _mode_fill(series: pd.Series) -> float:
        mode = series.dropna().mode()
        return -1.0 if mode.empty else float(mode.iloc[0])
