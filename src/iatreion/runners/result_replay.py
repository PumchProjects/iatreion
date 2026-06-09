from collections import defaultdict
from pathlib import Path
from shutil import copyfile

from iatreion.configs import ResultReplayConfig
from iatreion.models import ResultReplayModel
from iatreion.train_utils.fusion import (
    AvailableFusionArtifact,
    get_fold_fusion_artifact_path,
    get_run_fusion_artifact_path,
)
from iatreion.train_utils.results import ResultBundle
from iatreion.trainers.manifest import now_utc, write_manifest
from iatreion.trainers.recorder import Finish, Recorder, TrainerReturn
from iatreion.trainers.utils import log_available_fusion_artifact


class ResultReplayRunner:
    def __init__(self, config: ResultReplayConfig) -> None:
        self.config = config
        self.model = ResultReplayModel(config)
        self.train = config.train
        self.names = self.model.names
        self.objectives: dict[str, float] = {}

    def _fit_artifact(self, bundle: ResultBundle) -> AvailableFusionArtifact:
        return AvailableFusionArtifact.fit(
            self.train,
            bundle.names,
            bundle.y_true,
            bundle.y_pos_score_list,
            bundle.y_mask_list,
        )

    def _save_artifact(
        self, artifact: AvailableFusionArtifact, name: str = 'available_fusion'
    ) -> Path:
        path = get_run_fusion_artifact_path(self.train._log_dir)
        artifact.save(path)
        log_available_fusion_artifact(self.train, name, artifact)
        return path

    def _publish_final_artifact(self, source: Path) -> None:
        target = self.model.published_artifact_path
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)

    def _record_fused(
        self,
        recorders: dict[str, Recorder],
        bundle: ResultBundle,
        artifact: AvailableFusionArtifact,
    ) -> None:
        y_score = artifact.predict_scores(
            bundle.names,
            bundle.y_pos_score_list,
            bundle.y_mask_list,
        )
        thresholds: dict[str, float | None] = {'original': None} | artifact.thresholds
        for threshold_name, threshold in thresholds.items():
            recorder_name = f'all_calibrated_fusion_{threshold_name}'
            recorder = recorders[recorder_name]
            recorder.record(
                TrainerReturn(
                    0.0,
                    bundle.y_true,
                    y_score.copy(),
                    sample_id=bundle.sample_id,
                    outer_fold=bundle.outer_fold,
                    inner_fold=bundle.inner_fold,
                    kind=bundle.kind,
                    threshold=threshold,
                    test_mask=bundle.all_missing_mask,
                )
            )

    def _store_finish(self, name: str, recorder: Recorder) -> Finish:
        finish = recorder.finish(calc_ci=False)
        finish.log(name)
        for metric, value in finish.final.metrics.items():
            self.objectives[f'{name}/{metric}'] = value
        self.objectives[f'{name}/Time'] = finish.final.time
        return finish

    def _write_manifest(self, started_at: str) -> None:
        write_manifest(
            self.config,
            started_at=started_at,
            objectives=self.objectives,
            parameter_map={},
        )

    def _run_final(self) -> None:
        store = self.model.store
        bundle = store.bundle(self.names)
        artifact = self._fit_artifact(bundle)
        artifact_path = self._save_artifact(artifact)
        self._publish_final_artifact(artifact_path)
        recorders: defaultdict[str, Recorder] = defaultdict(
            lambda: Recorder(self.train)
        )
        self._record_fused(recorders, bundle, artifact)
        for name, recorder in recorders.items():
            self._store_finish(name, recorder)

    def _run_internal(self) -> None:
        store = self.model.store
        fold_recorders: defaultdict[str, Recorder] = defaultdict(
            lambda: Recorder(self.train)
        )
        for outer_fold in store.outer_folds(self.names[0]):
            inner = store.bundle(self.names, suffix=f'_inner_{outer_fold}')
            artifact = self._fit_artifact(inner)
            artifact.save(
                get_fold_fusion_artifact_path(self.train._log_dir, outer_fold)
            )
            log_available_fusion_artifact(
                self.train,
                f'weights_available_fusion_outer_{outer_fold}',
                artifact,
            )
            outer = store.bundle(self.names, outer_fold=outer_fold)
            self._record_fused(fold_recorders, outer, artifact)

        global_artifact = self._fit_artifact(store.bundle(self.names))
        self._save_artifact(global_artifact)
        for name, recorder in fold_recorders.items():
            self._store_finish(name, recorder)

    def run(self) -> None:
        self.train._log_dir = self.model.output_root
        self.train._log_dir.mkdir(parents=True, exist_ok=True)
        started_at = now_utc()
        with self.train.logging('train'):
            if self.train.final:
                self._run_final()
            else:
                self._run_internal()
            self._write_manifest(started_at)
