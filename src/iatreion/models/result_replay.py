from dataclasses import dataclass
from pathlib import Path

from iatreion.configs import ResultReplayConfig
from iatreion.log_paths import (
    RESULT_REPLAY_DIR,
    final_calibration_model_root,
    final_calibration_root,
    final_model_root,
    training_model_root,
    training_root,
)
from iatreion.train_utils import get_data_names
from iatreion.train_utils.fusion import get_published_fusion_artifact_path
from iatreion.train_utils.results import ResultStore


@dataclass(frozen=True)
class ResultReplaySourceTarget:
    aggregate: str
    folder_name: str


class ResultReplayModel:
    def __init__(self, config: ResultReplayConfig) -> None:
        self.config = config

    @property
    def names(self) -> list[str]:
        return list(self.config.eval_names or self.config.dataset.names)

    @property
    def subset_key(self) -> str:
        return '_'.join(self.names)

    @property
    def final_source_target(self) -> ResultReplaySourceTarget:
        names = get_data_names(self.config.dataset, self.config.train)
        if names == ['all_concat']:
            return ResultReplaySourceTarget(
                aggregate='concat',
                folder_name='all_concat',
            )
        return ResultReplaySourceTarget(
            aggregate='average',
            folder_name=self.config.train.eval_name_str or self.config.dataset.name_str,
        )

    @property
    def source_aggregate(self) -> str:
        if self.config.train.final:
            return self.final_source_target.aggregate
        return self.config.train.aggregate

    @property
    def source_root(self) -> Path:
        train = self.config.train
        if train.final:
            target = self.final_source_target
            return (
                final_calibration_model_root(
                    train.log_root,
                    self.config.dataset.name_str,
                    train.group_name_str,
                    self.config.source_model,
                    target.aggregate,
                )
                / target.folder_name
            )
        return training_model_root(
            train.log_root,
            self.config.dataset.name_str,
            train.group_name_str,
            self.config.source_model,
            train.aggregate,
        )

    @property
    def result_replay_root(self) -> Path:
        return (
            training_root(
                self.config.train.log_root,
                self.config.dataset.name_str,
                self.config.train.group_name_str,
            )
            / RESULT_REPLAY_DIR
        )

    @property
    def output_root(self) -> Path:
        train = self.config.train
        if train.final:
            target = self.final_source_target
            return (
                final_calibration_root(
                    train.log_root,
                    self.config.dataset.name_str,
                    train.group_name_str,
                )
                / RESULT_REPLAY_DIR
                / self.config.source_model
                / target.aggregate
                / self.subset_key
            )
        return (
            self.result_replay_root
            / self.config.source_model
            / train.aggregate
            / self.subset_key
        )

    @property
    def published_artifact_path(self) -> Path:
        train = self.config.train
        return get_published_fusion_artifact_path(
            final_model_root(
                train.log_root,
                train.group_name_str,
                self.config.source_model,
            ),
            self.names,
        )

    @property
    def store(self) -> ResultStore:
        return ResultStore(self.source_root)
