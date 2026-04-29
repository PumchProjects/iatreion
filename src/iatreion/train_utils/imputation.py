from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from iatreion.exceptions import IatreionException
from iatreion.utils import load_dict, save_dict

SIMPLE_IMPUTER_SUFFIX = '.simple-imputer.toml'


def get_simple_imputer_path(rrl_path: Path) -> Path:
    return rrl_path.with_name(f'{rrl_path.stem}{SIMPLE_IMPUTER_SUFFIX}')


@dataclass(frozen=True)
class SimpleImputerColumn:
    name: str
    fill_value: float
    snap_upper: int | None = None


@dataclass(frozen=True)
class SimpleImputerArtifact:
    columns: list[SimpleImputerColumn]
    version: int = 1
    strategy: str = 'simple'

    @classmethod
    def load(cls, path: Path) -> 'SimpleImputerArtifact':
        if not path.is_file():
            raise IatreionException(
                'Simple imputer artifact not found: $path. '
                'Retrain the original RRL model to export imputation statistics.',
                path=str(path),
            )
        data = load_dict(path)
        return cls(
            version=int(data['version']),
            strategy=str(data['strategy']),
            columns=[
                SimpleImputerColumn(
                    name=str(column['name']),
                    fill_value=float(column['fill_value']),
                    snap_upper=(
                        None
                        if column.get('snap_upper') is None
                        else int(column['snap_upper'])
                    ),
                )
                for column in data['columns']
            ],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: list[dict[str, object]] = []
        for column in self.columns:
            data: dict[str, object] = {
                'name': column.name,
                'fill_value': column.fill_value,
            }
            if column.snap_upper is not None:
                data['snap_upper'] = column.snap_upper
            columns.append(data)
        save_dict(
            {
                'version': self.version,
                'strategy': self.strategy,
                'columns': columns,
            },
            path,
        )

    def apply(
        self,
        data: pd.DataFrame,
        *,
        preserve_all_missing: bool = True,
    ) -> pd.DataFrame:
        frame = data.copy()
        if frame.empty:
            return frame

        column_names = [column.name for column in self.columns]
        available = (
            ~frame[column_names].isna().all(axis=1)
            if preserve_all_missing
            else pd.Series(True, index=frame.index)
        )
        if not available.any():
            return frame

        rows = frame.index[available]
        for column in self.columns:
            frame.loc[rows, column.name] = frame.loc[rows, column.name].fillna(
                column.fill_value
            )
            if column.snap_upper is None:
                continue
            values = frame.loc[rows, column.name].to_numpy(dtype=float, copy=True)
            valid = ~np.isnan(values)
            values[valid] = np.clip(np.floor(values[valid] + 0.5), 0, column.snap_upper)
            frame.loc[rows, column.name] = values
        return frame
