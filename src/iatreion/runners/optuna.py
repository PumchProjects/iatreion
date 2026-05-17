from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from shutil import copyfile
from threading import Lock
from typing import Any, Literal, override

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.storages import RDBStorage
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.train_utils import (
    FoldSpec,
    get_cv_fold_specs,
    get_data_names,
    get_nested_fold_specs,
    get_train_test,
    read_data,
)
from iatreion.train_utils.fusion import FUSION_ARTIFACT_FILE
from iatreion.trainers import ModelTrainer
from iatreion.utils import (
    apply_overrides,
    disable_progress,
    load_dict,
    logger,
    progress,
    save_dict,
    suppress_console_logs,
)

from .base import Runner

type SearchSpaceKind = Literal['float', 'int', 'categorical']
type StudyDirection = Literal['maximize', 'minimize']
type SamplerName = Literal['tpe']
type PrunerName = Literal['none']


@dataclass(frozen=True)
class SearchSpace:
    kind: SearchSpaceKind
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SearchSpace':
        return cls(
            kind=data['type'],
            low=data.get('low'),
            high=data.get('high'),
            step=data.get('step'),
            log=data.get('log', False),
            choices=data.get('choices'),
        )

    def sample(self, trial: Trial, name: str) -> Any:
        match self.kind:
            case 'float':
                assert self.low is not None and self.high is not None
                if self.step is not None:
                    return trial.suggest_float(
                        name, float(self.low), float(self.high), step=float(self.step)
                    )
                return trial.suggest_float(
                    name, float(self.low), float(self.high), log=self.log
                )
            case 'int':
                assert self.low is not None and self.high is not None
                return trial.suggest_int(
                    name,
                    int(self.low),
                    int(self.high),
                    step=1 if self.step is None else int(self.step),
                    log=self.log,
                )
            case 'categorical':
                assert self.choices is not None
                return trial.suggest_categorical(name, self.choices)
            case kind:
                raise ValueError(f'Unknown search-space kind: {kind}!')


@dataclass(frozen=True)
class TuningStudyConfig:
    name: str
    objective: str
    direction: StudyDirection = 'maximize'
    n_trials: int | None = None
    timeout_sec: int | None = None
    sampler: SamplerName = 'tpe'
    seed: int = 42
    n_startup_trials: int = 20
    multivariate: bool = True
    pruner: PrunerName = 'none'
    load_if_exists: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TuningStudyConfig':
        return cls(
            name=data.get('name', ''),
            objective=data['objective'],
            direction=data.get('direction', 'maximize'),
            n_trials=data.get('n_trials'),
            timeout_sec=data.get('timeout_sec'),
            sampler=data.get('sampler', 'tpe'),
            seed=data.get('seed', 42),
            n_startup_trials=data.get('n_startup_trials', 20),
            multivariate=data.get('multivariate', True),
            pruner=data.get('pruner', 'none'),
            load_if_exists=data.get('load_if_exists', True),
        )


@dataclass(frozen=True)
class TuningExecutionConfig:
    trial_log_root: Path = Path('logs_optuna')
    fail_value: float = 0.0
    n_jobs: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TuningExecutionConfig':
        return cls(
            trial_log_root=Path(data.get('trial_log_root', 'logs_optuna')),
            fail_value=float(data.get('fail_value', 0.0)),
            n_jobs=data.get('n_jobs'),
        )


@dataclass(frozen=True)
class TuningSpec:
    study: TuningStudyConfig
    execution: TuningExecutionConfig
    search: dict[str, SearchSpace]

    @classmethod
    def load(cls, config: ModelConfig) -> 'TuningSpec':
        assert (path := config.tune_config) is not None
        data = load_dict(path)
        if 'study' not in data:
            raise ValueError(f'Missing [study] section in tuning config: {path}')
        if 'search' not in data:
            raise ValueError(f'Missing [search] section in tuning config: {path}')

        study = TuningStudyConfig.from_dict(data['study'])
        name = config.study_name or study.name or default_study_name(config)
        study = TuningStudyConfig(
            name=sanitize_name(name),
            objective=study.objective,
            direction=study.direction,
            n_trials=study.n_trials,
            timeout_sec=study.timeout_sec,
            sampler=study.sampler,
            seed=study.seed,
            n_startup_trials=study.n_startup_trials,
            multivariate=study.multivariate,
            pruner=study.pruner,
            load_if_exists=study.load_if_exists,
        )
        return cls(
            study=study,
            execution=TuningExecutionConfig.from_dict(data.get('execution', {})),
            search=flatten_search_space(data['search']),
        )

    @property
    def study_root(self) -> Path:
        return self.execution.trial_log_root / self.study.name


class DevicePool:
    def __init__(self, n_devices: int) -> None:
        self.logical_devices = list(range(max(n_devices, 1)))
        self.index = 0
        self.lock = Lock()

    def assign(self) -> int | None:
        if not self.logical_devices:
            return None
        with self.lock:
            device = self.logical_devices[self.index % len(self.logical_devices)]
            self.index += 1

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(device)
                return device
        except Exception:
            return None
        return None


def sanitize_name(name: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)
    return safe.strip('_') or 'study'


def default_study_name(config: ModelConfig) -> str:
    return '__'.join(
        [
            config.dataset.name_str,
            config.train.group_name_str,
            config.train.aggregate,
        ]
    )


def flatten_search_space(
    data: dict[str, Any], prefix: str = ''
) -> dict[str, SearchSpace]:
    search: dict[str, SearchSpace] = {}
    for key, value in data.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict) and 'type' in value:
            search[name] = SearchSpace.from_dict(value)
            continue
        if not isinstance(value, dict):
            raise ValueError(f'Invalid search-space entry for {name!r}: {value!r}')
        search |= flatten_search_space(value, name)
    return search


def dump_trial_info(
    root: Path,
    *,
    status: str,
    sampled: dict[str, Any],
    objectives: dict[str, float] | None = None,
    error: str | None = None,
) -> None:
    data: dict[str, Any] = {'status': status, 'sampled_params': sampled}
    if error is not None:
        data['error'] = error
    if objectives is not None:
        data['objectives'] = {
            key: value if isfinite(value) else 'nan'
            for key, value in objectives.items()
        }
    save_dict(data, root / 'trial_info.toml')


def append_study_log(root: Path, message: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec='seconds')
    with (root / 'study.log').open('a', encoding='utf-8') as file:
        print(f'{stamp} {message}', file=file)


def format_objective(value: float | None) -> str:
    if value is None:
        return 'NA'
    return f'{value:.6f}' if isfinite(value) else 'nan'


def study_label(root: Path, candidate: str) -> str:
    if len(root.parts) >= 3 and root.parts[-3] == 'nested':
        return f'nested {root.parts[-2]} {candidate}'
    if len(root.parts) >= 2 and root.parts[-2] == 'final':
        return f'final {candidate}'
    return candidate


class OptunaRunner(Runner):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        super().__init__(model_cls, config)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.spec = TuningSpec.load(config)
        self.device_pool = DevicePool(len(config.train.device_ids))

    def _get_sampler(self) -> BaseSampler:
        match self.spec.study.sampler:
            case 'tpe':
                return TPESampler(
                    seed=self.spec.study.seed,
                    n_startup_trials=self.spec.study.n_startup_trials,
                    multivariate=self.spec.study.multivariate,
                )
            case sampler:
                raise ValueError(f'Unknown Optuna sampler: {sampler}!')

    def _get_pruner(self) -> BasePruner:
        match self.spec.study.pruner:
            case 'none':
                return NopPruner()
            case pruner:
                raise ValueError(f'Unknown Optuna pruner: {pruner}!')

    def _create_study(self, root: Path, name: str) -> Study:
        root.mkdir(parents=True, exist_ok=True)
        storage = RDBStorage(
            url=f'sqlite:///{root / "study.db"}',
            engine_kwargs={'connect_args': {'timeout': 60}},
        )
        return optuna.create_study(
            study_name=name,
            storage=storage,
            direction=self.spec.study.direction,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
            load_if_exists=self.spec.study.load_if_exists,
        )

    def _training_config(
        self,
        overrides: dict[str, Any],
        *,
        folder_name: str | None = None,
        file_name: str = 'train.log',
    ) -> ModelConfig:
        full_overrides = {
            'tune_config': None,
            'study_name': None,
            'train.log_root': self.base_config.train.log_root,
        } | overrides
        config = apply_overrides(
            self.base_config,
            full_overrides,
        )
        config.register_log_dir(
            self.model_name, folder_name=folder_name, file_name=file_name
        )
        return config

    def _sample(self, trial: Trial) -> dict[str, Any]:
        return {
            key: space.sample(trial, key) for key, space in self.spec.search.items()
        }

    def _objective_key(self, candidate: str) -> str:
        objective = self.spec.study.objective
        if '/' in objective:
            return objective
        return f'{candidate}/{objective}'

    def _candidate_names(self) -> list[str]:
        return get_data_names(self.base_config.dataset, self.base_config.train)

    def _candidate_tuning_overrides(
        self, candidate: str, fold_specs: list[FoldSpec]
    ) -> dict[str, Any]:
        if candidate == 'all_concat':
            aggregate = 'concat'
            eval_names: list[str] = []
        else:
            aggregate = 'average'
            eval_names = [candidate]
        return {
            'importance_methods': [],
            'train.aggregate': aggregate,
            'train.eval_names': eval_names,
            'train.final': False,
            'train.n_outer_splits': len(fold_specs),
        }

    def _n_jobs(self) -> int:
        n_jobs = self.spec.execution.n_jobs
        if n_jobs is None:
            return 1
        if n_jobs < 1:
            raise ValueError('execution.n_jobs must be >= 1.')
        if n_jobs > 1:
            raise ValueError(
                'Parallel Optuna threads are disabled for SQLite studies. '
                'Use execution.n_jobs = 1 and reduce n_trials/folds for short runs.'
            )
        return n_jobs

    @staticmethod
    def _mark_unfinished_trials_failed(study: Study) -> None:
        unfinished = study.get_trials(
            deepcopy=False,
            states=(TrialState.RUNNING, TrialState.WAITING),
        )
        for trial in unfinished:
            study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
            logger.warning(
                'Marked stale Optuna trial #%s as FAIL in study "%s"',
                trial.number,
                study.study_name,
            )

    def _remaining_trials(self, study: Study) -> int | None:
        n_trials = self.spec.study.n_trials
        if n_trials is None:
            return None
        complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        return max(n_trials - complete, 0)

    @staticmethod
    def _complete_trials(study: Study) -> int:
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))

    @staticmethod
    def _trial_status(root: Path, trial: FrozenTrial) -> str:
        info_path = root / f'trial_{trial.number:04d}' / 'trial_info.toml'
        if not info_path.exists():
            return trial.state.name.lower()
        try:
            info = load_dict(info_path)
        except Exception:
            return trial.state.name.lower()
        status = info.get('status')
        return status if isinstance(status, str) else trial.state.name.lower()

    def _trial_callback(
        self,
        *,
        root: Path,
        candidate: str,
        task_id: int | None,
    ):
        label = study_label(root, candidate)

        def callback(study: Study, trial: FrozenTrial) -> None:
            complete = self._complete_trials(study)
            best = format_objective(study.best_value) if complete else 'NA'
            value = format_objective(trial.value)
            status = self._trial_status(root, trial)
            message = (
                f'{label} trial #{trial.number} {status}: value={value} best={best}'
            )
            logger.info(message)
            append_study_log(root, message)
            if task_id is not None:
                progress.update(
                    task_id,
                    completed=complete,
                    description=f'{label}: best={best}',
                )

        return callback

    def _run_trial(
        self,
        trial: Trial,
        *,
        candidate: str,
        fold_specs: list[FoldSpec],
        overrides: dict[str, Any],
        study_root: Path,
    ) -> float:
        sampled = self._sample(trial)
        trial_log_root = study_root / f'trial_{trial.number:04d}'
        config = self._training_config(
            overrides | sampled | {'train.log_root': trial_log_root},
            file_name=f'trial_{trial.number:04d}.log',
        )

        model: Model | None = None
        self.device_pool.assign()
        try:
            with suppress_console_logs(), disable_progress():
                model = self.model_cls(config)
                trainer = ModelTrainer(model, fold_specs=fold_specs, calc_ci=False)
                summary = trainer.train()
        except Exception as error:
            dump_trial_info(
                trial_log_root,
                status='failed',
                sampled=sampled,
                error=repr(error),
            )
            logger.exception(f'Optuna trial {trial.number} failed')
            return self.spec.execution.fail_value
        finally:
            if model is not None:
                model.close()
            config.close_log_handler()

        objective_name = self._objective_key(candidate)
        objective = summary.objectives.get(objective_name)
        if objective is None:
            objectives = {
                key: float(value) for key, value in summary.objectives.items()
            }
            dump_trial_info(
                trial_log_root,
                status=f'missing-objective:{objective_name}',
                sampled=sampled,
                objectives=objectives,
            )
            return self.spec.execution.fail_value
        objective_value = float(objective)
        objectives = {key: float(value) for key, value in summary.objectives.items()}
        if not isfinite(objective_value):
            dump_trial_info(
                trial_log_root,
                status=f'nonfinite-objective:{objective_name}',
                sampled=sampled,
                objectives=objectives,
            )
            return self.spec.execution.fail_value

        dump_trial_info(
            trial_log_root,
            status='completed',
            sampled=sampled,
            objectives=objectives,
        )
        return objective_value

    def _run_study(
        self,
        *,
        candidate: str,
        fold_specs: list[FoldSpec],
        root: Path,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        study = self._create_study(root, sanitize_name(candidate))
        label = study_label(root, candidate)
        objective_name = self._objective_key(candidate)
        message = f'Optuna study {label} started: objective={objective_name}'
        logger.info(message)
        append_study_log(root, f'{message} root={root}')
        self._mark_unfinished_trials_failed(study)
        remaining_trials = self._remaining_trials(study)
        if remaining_trials != 0:
            task_id = progress.add_task(
                f'{label}: best=NA',
                total=self.spec.study.n_trials,
                completed=self._complete_trials(study),
            )
            try:
                study.optimize(
                    lambda trial: self._run_trial(
                        trial,
                        candidate=candidate,
                        fold_specs=fold_specs,
                        overrides=overrides,
                        study_root=root,
                    ),
                    n_trials=remaining_trials,
                    timeout=self.spec.study.timeout_sec,
                    n_jobs=self._n_jobs(),
                    callbacks=[
                        self._trial_callback(
                            root=root,
                            candidate=candidate,
                            task_id=task_id,
                        )
                    ],
                )
            finally:
                progress.remove_task(task_id)
        else:
            message = f'Optuna study {label} already has enough complete trials'
            logger.info(message)
            append_study_log(root, message)
        trial_info = root / f'trial_{study.best_trial.number:04d}' / 'trial_info.toml'
        info = {
            'trial_number': study.best_trial.number,
            'value': study.best_value,
            'params': study.best_trial.params,
            'trial_info': str(trial_info),
        }
        save_dict(info, root / 'best_trial.toml')
        save_dict(study.best_trial.params, root / 'params.toml')
        message = (
            f'Best {label} trial #{study.best_trial.number}: '
            f'value={study.best_value:.6f}'
        )
        logger.info(message)
        append_study_log(root, f'{message} params={study.best_trial.params}')
        return dict(study.best_trial.params)

    def _nested_tune(
        self,
    ) -> tuple[dict[int, dict[str, dict[str, Any]]], list[FoldSpec]]:
        _, _, ref_y, _ = read_data(self.base_config.dataset, self.base_config.train)
        candidates = self._candidate_names()
        selected: dict[int, dict[str, dict[str, Any]]] = {}

        for outer_fold, (train_outer, _test_outer) in enumerate(
            get_train_test(self.base_config.train.n_outer_splits, ref_y)
        ):
            fold_specs = get_cv_fold_specs(
                self.base_config.train.n_inner_splits,
                ref_y,
                pool_index=train_outer,
            )
            selected[outer_fold] = {}
            for candidate in candidates:
                root = (
                    self.spec.study_root
                    / 'nested'
                    / f'outer_{outer_fold}'
                    / sanitize_name(candidate)
                )
                overrides = self._candidate_tuning_overrides(candidate, fold_specs)
                selected[outer_fold][candidate] = self._run_study(
                    candidate=candidate,
                    fold_specs=fold_specs,
                    root=root,
                    overrides=overrides,
                )

        save_dict(
            {f'outer_{outer}': params for outer, params in selected.items()},
            self.spec.study_root / 'nested' / 'selected_params.toml',
        )
        return selected, get_nested_fold_specs(self.base_config.train, ref_y)

    def _evaluate_nested(self) -> None:
        selected, fold_specs = self._nested_tune()
        config = self._training_config({'tune_config': None})
        model: Model | None = None
        try:
            model = self.model_cls(config)
            ModelTrainer(model, fold_specs=fold_specs, parameter_map=selected).train()
        finally:
            if model is not None:
                model.close()
            config.close_log_handler()

    def _final_tune(self) -> dict[str, dict[str, Any]]:
        train = self.base_config.train
        _, _, ref_y, _ = read_data(self.base_config.dataset, train)
        fold_specs = get_cv_fold_specs(train.n_inner_splits, ref_y)
        selected: dict[str, dict[str, Any]] = {}
        for candidate in self._candidate_names():
            root = self.spec.study_root / 'final' / sanitize_name(candidate)
            overrides = self._candidate_tuning_overrides(candidate, fold_specs)
            selected[candidate] = self._run_study(
                candidate=candidate,
                fold_specs=fold_specs,
                root=root,
                overrides=overrides,
            )
        save_dict(selected, self.spec.study_root / 'final' / 'selected_params.toml')
        return selected

    def _fit_final_fusion_artifact(self, selected: dict[str, dict[str, Any]]) -> Path:
        train = self.base_config.train
        _, _, ref_y, _ = read_data(self.base_config.dataset, train)
        fold_specs = get_cv_fold_specs(train.n_inner_splits, ref_y)
        candidates = self._candidate_names()
        artifact_aggregate = 'concat' if candidates == ['all_concat'] else 'average'
        config = self._training_config(
            {
                'importance_methods': [],
                'train.aggregate': artifact_aggregate,
                'train.eval_names': [] if candidates == ['all_concat'] else candidates,
                'train.final': False,
                'train.n_outer_splits': len(fold_specs),
            },
            folder_name='final_calibration',
        )

        model: Model | None = None
        try:
            model = self.model_cls(config)
            ModelTrainer(
                model,
                fold_specs=fold_specs,
                parameter_map=selected,
                calc_ci=False,
            ).train()
        finally:
            if model is not None:
                model.close()
            config.close_log_handler()
        return config.train._log_dir / FUSION_ARTIFACT_FILE

    def _fit_final_models(
        self, selected: dict[str, dict[str, Any]], artifact: Path | None
    ) -> None:
        config = self._training_config({'train.final': True})
        if artifact is not None:
            target = config.train._log_dir / FUSION_ARTIFACT_FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(artifact, target)

        model: Model | None = None
        try:
            model = self.model_cls(config)
            ModelTrainer(model, parameter_map=selected).train()
        finally:
            if model is not None:
                model.close()
            config.close_log_handler()

    def _finalize(self) -> None:
        selected = self._final_tune()
        artifact = (
            self._fit_final_fusion_artifact(selected)
            if self.base_config.train.num_class == 2
            else None
        )
        self._fit_final_models(selected, artifact)

    @override
    def run(self) -> None:
        if self.base_config.train.final:
            self._finalize()
        else:
            self._evaluate_nested()
