from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from iatreion.exceptions import IatreionException


@dataclass(frozen=True)
class ResultRecord:
    sample_id: NDArray
    y_true: NDArray
    y_pos_score: NDArray
    y_mask: NDArray
    outer_fold: NDArray
    inner_fold: NDArray
    kind: NDArray

    @classmethod
    def load(cls, path: Path) -> 'ResultRecord':
        if not path.is_file():
            raise IatreionException('Result NPZ not found: $path', path=str(path))
        data = np.load(path)
        return cls(
            sample_id=data['sample_id'].astype(str),
            y_true=data['y_true'],
            y_pos_score=data['y_score'][:, 1],
            y_mask=data['y_mask'].astype(bool),
            outer_fold=data['outer_fold'].astype(int),
            inner_fold=data['inner_fold'].astype(int),
            kind=data['kind'].astype(str),
        )

    def select_outer(self, outer_fold: int) -> 'ResultRecord':
        return self[self.outer_fold == outer_fold]

    def __getitem__(self, index: NDArray) -> 'ResultRecord':
        return ResultRecord(
            sample_id=self.sample_id[index],
            y_true=self.y_true[index],
            y_pos_score=self.y_pos_score[index],
            y_mask=self.y_mask[index],
            outer_fold=self.outer_fold[index],
            inner_fold=self.inner_fold[index],
            kind=self.kind[index],
        )


@dataclass(frozen=True)
class ResultBundle:
    names: list[str]
    sample_id: NDArray
    y_true: NDArray
    y_pos_score_list: list[NDArray]
    y_mask_list: list[NDArray]
    outer_fold: NDArray
    inner_fold: NDArray
    kind: NDArray

    @property
    def all_missing_mask(self) -> NDArray:
        return np.column_stack(self.y_mask_list).astype(bool).all(axis=1)


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, name: str, *, suffix: str = '') -> ResultRecord:
        return ResultRecord.load(self.root / f'results_{name}{suffix}.npz')

    @staticmethod
    def _align(
        reference: ResultRecord, record: ResultRecord
    ) -> tuple[NDArray, NDArray]:
        index = {sample_id: i for i, sample_id in enumerate(record.sample_id)}
        positions = np.array([index[sample_id] for sample_id in reference.sample_id])
        return record.y_pos_score[positions], record.y_mask[positions]

    def bundle(
        self,
        names: list[str],
        *,
        suffix: str = '',
        outer_fold: int | None = None,
    ) -> ResultBundle:
        records = [self.load(name, suffix=suffix) for name in names]
        if outer_fold is not None:
            records = [record.select_outer(outer_fold) for record in records]
        reference = records[0]
        y_pos_score_list = []
        y_mask_list = []
        for record in records:
            y_pos_score, y_mask = self._align(reference, record)
            y_pos_score_list.append(y_pos_score)
            y_mask_list.append(y_mask)
        return ResultBundle(
            names=names,
            sample_id=reference.sample_id,
            y_true=reference.y_true,
            y_pos_score_list=y_pos_score_list,
            y_mask_list=y_mask_list,
            outer_fold=reference.outer_fold,
            inner_fold=reference.inner_fold,
            kind=reference.kind,
        )

    def outer_folds(self, name: str) -> list[int]:
        return sorted(np.unique(self.load(name).outer_fold).astype(int).tolist())
