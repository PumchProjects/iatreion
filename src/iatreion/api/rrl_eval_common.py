from typing import overload

import numpy as np
import pandas as pd


@overload
def get_max_label(arr: list[float], labels: list[str]) -> str: ...


@overload
def get_max_label(arr: pd.DataFrame) -> 'pd.Series[str]': ...


def get_max_label(
    arr: list[float] | pd.DataFrame,
    labels: list[str] | None = None,
) -> 'str | pd.Series[str]':
    if isinstance(arr, list):
        assert labels is not None
        return labels[np.argmax(arr).item()]
    max_labels = arr.fillna(0).idxmax(axis=1, skipna=False).astype(str)
    max_labels.loc[arr.isna().all(axis=1)] = ''
    return max_labels


@overload
def calc_score(arr: list[float]) -> float: ...


@overload
def calc_score(arr: pd.DataFrame) -> 'pd.Series[float]': ...


def calc_score(arr: list[float] | pd.DataFrame) -> 'float | pd.Series[float]':
    if isinstance(arr, list):
        # Weights and biases are original values
        return max(arr) - min(arr)
    # Predictions are probabilities
    return arr.max(axis=1)


def deduplicate_by_keep(df: pd.DataFrame, keep: str) -> pd.DataFrame:
    return df[~df.index.duplicated(keep=keep)]


def calc_signed_score(
    weights: list[float],
    labels: list[str],
    target_label: str,
) -> float:
    if not target_label or len(labels) != 2 or target_label not in labels:
        return float('nan')
    target_idx = labels.index(target_label)
    other_idx = 1 - target_idx
    return weights[target_idx] - weights[other_idx]


def series_item(series: pd.Series) -> float:
    return float(series.item())


def probability_for_label(row: pd.Series, label: str) -> float:
    value = row.get(label, np.nan)
    return float(value) if pd.notna(value) else float('nan')


def opposing_label(labels: tuple[str, ...], target_label: str) -> str:
    for label in labels:
        if label != target_label:
            return label
    return f'not {target_label}'
