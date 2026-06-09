import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from importlib import import_module
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal, override

import optuna
from optuna.exceptions import ExperimentalWarning
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
from iatreion.trainers import ModelTrainer
from iatreion.utils import (
    apply_overrides,
    disable_progress,
    load_dict,
    logger,
    progress,
    save_dict,
    suppress_console_logs,
    task,
)

from .base import Runner, model_name_for
from .final_calibration import fit_final_fusion_artifact, publish_fusion_artifact

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
            n_trials=data.get('n-trials'),
            timeout_sec=data.get('timeout-sec'),
            sampler=data.get('sampler', 'tpe'),
            seed=data.get('seed', 42),
            n_startup_trials=data.get('n-startup-trials', 20),
            multivariate=data.get('multivariate', True),
            pruner=data.get('pruner', 'none'),
            load_if_exists=data.get('load-if-exists', True),
        )


@dataclass(frozen=True)
class TuningExecutionConfig:
    trial_log_root: Path = Path('logs_optuna')
    fail_value: float = 0.0
    n_jobs: int | None = None
    study_workers: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TuningExecutionConfig':
        return cls(
            trial_log_root=Path(data.get('trial-log-root', 'logs_optuna')),
            fail_value=float(data.get('fail-value', 0.0)),
            n_jobs=data.get('n-jobs'),
            study_workers=data.get('study-workers'),
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


@dataclass(frozen=True)
class TuningTarget:
    name: str
    aggregate: str
    eval_names: list[str]

    @classmethod
    def from_candidate(cls, candidate: str) -> 'TuningTarget':
        if candidate == 'all_concat':
            return cls(name=candidate, aggregate='concat', eval_names=[])
        return cls(name=candidate, aggregate='average', eval_names=[candidate])

    def training_overrides(self, n_folds: int) -> dict[str, Any]:
        return {
            'importance_methods': [],
            'train.aggregate': self.aggregate,
            'train._eval_names': self.eval_names,
            'train.final': False,
            'train.n_outer_splits': n_folds,
        }


@dataclass(frozen=True)
class StudyJob:
    target: TuningTarget
    fold_specs: list[FoldSpec]
    root: Path
    device_id: int | None = None

    @property
    def candidate(self) -> str:
        return self.target.name


@dataclass(frozen=True)
class StudyResult:
    candidate: str
    params: dict[str, Any]


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
            search[name.replace('-', '_')] = SearchSpace.from_dict(value)
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


def make_training_config(
    base_config: ModelConfig,
    model_name: str,
    overrides: dict[str, Any],
    *,
    folder_name: str | None = None,
    file_name: str = 'train.log',
) -> ModelConfig:
    config = apply_overrides(
        base_config,
        {
            'tune_config': None,
            'study_name': None,
            'train.log_root': base_config.train.log_root,
        }
        | overrides,
    )
    if folder_name is None:
        folder_name = config.log_folder_name
    config.register_log_dir(model_name, folder_name=folder_name, file_name=file_name)
    return config


def class_path(cls: type) -> str:
    return f'{cls.__module__}:{cls.__qualname__}'


def import_class(path: str) -> type[Model]:
    module_name, _, qualname = path.partition(':')
    cls: Any = import_module(module_name)
    for name in qualname.split('.'):
        cls = getattr(cls, name)
    return cls


def init_study_worker(device_id: int | None) -> None:
    if device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_study_worker(
    model_cls_path: str,
    config: ModelConfig,
    job: StudyJob,
) -> StudyResult:
    if job.device_id is not None:
        config.train.device_id = str(job.device_id)
    model_cls = import_class(model_cls_path)
    executor = OptunaStudyExecutor(model_cls, config, TuningSpec.load(config))
    with suppress_console_logs(logging.CRITICAL + 1), disable_progress():
        params = executor.run(job, show_progress=False)
    config.close_log_handler()
    return StudyResult(candidate=job.candidate, params=params)


class OptunaStudyExecutor:
    def __init__(
        self,
        model_cls: type[Model],
        base_config: ModelConfig,
        spec: TuningSpec,
    ) -> None:
        self.model_cls = model_cls
        self.base_config = base_config
        self.spec = spec
        self.model_name = model_name_for(model_cls)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _get_sampler(self) -> BaseSampler:
        match self.spec.study.sampler:
            case 'tpe':
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ExperimentalWarning)
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
        file_name: str = 'train.log',
    ) -> ModelConfig:
        return make_training_config(
            self.base_config,
            self.model_name,
            overrides,
            file_name=file_name,
        )

    def _sample(self, trial: Trial) -> dict[str, Any]:
        return {
            key: space.sample(trial, key) for key, space in self.spec.search.items()
        }

    def _objective_key(self, target: TuningTarget) -> str:
        objective = self.spec.study.objective
        if '/' in objective:
            return objective
        return f'{target.name}/{objective}'

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
        target: TuningTarget,
        task_id: int | None,
    ):
        label = study_label(root, target.name)

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
        job: StudyJob,
    ) -> float:
        sampled = self._sample(trial)
        trial_log_root = job.root / f'trial_{trial.number:04d}'
        config = self._training_config(
            job.target.training_overrides(len(job.fold_specs))
            | sampled
            | {'train.log_root': trial_log_root},
            file_name=f'trial_{trial.number:04d}.log',
        )

        model: Model | None = None
        try:
            with suppress_console_logs(), disable_progress():
                model = self.model_cls(config)
                trainer = ModelTrainer(
                    model,
                    fold_specs=job.fold_specs,
                    calc_ci=False,
                )
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

        objective_name = self._objective_key(job.target)
        objective = summary.objectives.get(objective_name)
        objectives = {key: float(value) for key, value in summary.objectives.items()}
        if objective is None:
            dump_trial_info(
                trial_log_root,
                status=f'missing-objective:{objective_name}',
                sampled=sampled,
                objectives=objectives,
            )
            return self.spec.execution.fail_value

        objective_value = float(objective)
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

    def run(self, job: StudyJob, *, show_progress: bool = True) -> dict[str, Any]:
        study = self._create_study(job.root, sanitize_name(job.candidate))
        label = study_label(job.root, job.candidate)
        objective_name = self._objective_key(job.target)
        message = f'Optuna study "{label}" started: objective="{objective_name}"'
        logger.info(message)
        append_study_log(job.root, f'{message} root={job.root}')
        self._mark_unfinished_trials_failed(study)

        remaining_trials = self._remaining_trials(study)
        if remaining_trials != 0:
            task_id = (
                progress.add_task(
                    f'{label}: best=NA',
                    total=self.spec.study.n_trials,
                    completed=self._complete_trials(study),
                )
                if show_progress
                else None
            )
            try:
                study.optimize(
                    lambda trial: self._run_trial(trial, job=job),
                    n_trials=remaining_trials,
                    timeout=self.spec.study.timeout_sec,
                    n_jobs=self._n_jobs(),
                    callbacks=[
                        self._trial_callback(
                            root=job.root,
                            target=job.target,
                            task_id=task_id,
                        )
                    ],
                )
            finally:
                if task_id is not None:
                    progress.remove_task(task_id)
        else:
            message = f'Optuna study {label} already has enough complete trials'
            logger.info(message)
            append_study_log(job.root, message)

        trial_info = (
            job.root / f'trial_{study.best_trial.number:04d}' / 'trial_info.toml'
        )
        info = {
            'trial_number': study.best_trial.number,
            'value': study.best_value,
            'params': study.best_trial.params,
            'trial_info': str(trial_info),
        }
        save_dict(info, job.root / 'best_trial.toml')
        save_dict(study.best_trial.params, job.root / 'params.toml')
        message = (
            f'Best {label} trial #{study.best_trial.number}: '
            f'value={study.best_value:.6f}'
        )
        logger.info(message)
        append_study_log(job.root, f'{message} params={study.best_trial.params}')
        return dict(study.best_trial.params)


class OptunaRunner(Runner):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        super().__init__(model_cls, config)
        self.spec = TuningSpec.load(config)
        self.executor = OptunaStudyExecutor(model_cls, config, self.spec)

    def _training_config(self, overrides: dict[str, Any]) -> ModelConfig:
        return make_training_config(self.base_config, self.model_name, overrides)

    def _targets(self) -> list[TuningTarget]:
        return [
            TuningTarget.from_candidate(candidate)
            for candidate in get_data_names(
                self.base_config.dataset,
                self.base_config.train,
            )
        ]

    def _study_worker_count(self, n_studies: int) -> int:
        study_workers = self.spec.execution.study_workers
        if study_workers is not None:
            if study_workers < 1:
                raise ValueError('execution.study_workers must be >= 1.')
            return min(study_workers, n_studies)
        return min(max(len(self.base_config.train.device_ids), 1), n_studies)

    def _study_worker_devices(self, workers: int) -> list[int | None]:
        devices = self.base_config.train.device_ids
        if not devices:
            return [None] * workers
        return [devices[index % len(devices)] for index in range(workers)]

    def _run_study_jobs(self, jobs: list[StudyJob]) -> dict[str, dict[str, Any]]:
        if not jobs:
            return {}

        workers = self._study_worker_count(len(jobs))
        if workers == 1:
            return {job.candidate: self.executor.run(job) for job in jobs}

        devices = self._study_worker_devices(workers)
        device_label = ', '.join(
            'CPU' if device is None else str(device) for device in devices
        )
        logger.info(
            f'Running {len(jobs)} Optuna studies with {workers} workers '
            f'on devices: {device_label}'
        )
        context = get_context('spawn')
        model_cls_path = class_path(self.model_cls)
        executors = [
            ProcessPoolExecutor(
                max_workers=1,
                max_tasks_per_child=1,
                mp_context=context,
                initializer=init_study_worker,
                initargs=(device,),
            )
            for device in devices
        ]
        futures = {}
        try:
            for index, job in enumerate(jobs):
                slot = index % workers
                assigned = replace(job, device_id=devices[slot])
                futures[
                    executors[slot].submit(
                        run_study_worker,
                        model_cls_path,
                        self.base_config,
                        assigned,
                    )
                ] = assigned

            results: dict[str, dict[str, Any]] = {}
            with task('Study:', len(futures)) as study_advance:
                for future in as_completed(futures):
                    job = futures[future]
                    result = future.result()
                    results[result.candidate] = result.params
                    label = study_label(job.root, job.candidate)
                    logger.info(f'Optuna study "{label}" finished')
                    study_advance()
        finally:
            for executor in executors:
                executor.shutdown(cancel_futures=True)

        return {job.candidate: results[job.candidate] for job in jobs}

    def _nested_tune(
        self,
    ) -> tuple[dict[int, dict[str, dict[str, Any]]], list[FoldSpec]]:
        _, _, ref_y, _ = read_data(self.base_config.dataset, self.base_config.train)
        targets = self._targets()
        selected: dict[int, dict[str, dict[str, Any]]] = {}

        for outer_fold, (train_outer, _test_outer) in enumerate(
            get_train_test(self.base_config.train.n_outer_splits, ref_y)
        ):
            fold_specs = get_cv_fold_specs(
                self.base_config.train.n_inner_splits,
                ref_y,
                pool_index=train_outer,
            )
            jobs = [
                StudyJob(
                    target=target,
                    fold_specs=fold_specs,
                    root=(
                        self.spec.study_root
                        / 'nested'
                        / f'outer_{outer_fold}'
                        / sanitize_name(target.name)
                    ),
                )
                for target in targets
            ]
            selected[outer_fold] = self._run_study_jobs(jobs)

        save_dict(
            {f'outer_{outer}': params for outer, params in selected.items()},
            self.spec.study_root / 'nested' / 'selected_params.toml',
        )
        return selected, get_nested_fold_specs(self.base_config.train, ref_y)

    def _evaluate_nested(self) -> None:
        selected, fold_specs = self._nested_tune()
        config = self._training_config({})
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
        jobs = [
            StudyJob(
                target=target,
                fold_specs=fold_specs,
                root=self.spec.study_root / 'final' / sanitize_name(target.name),
            )
            for target in self._targets()
        ]
        selected = self._run_study_jobs(jobs)
        save_dict(selected, self.spec.study_root / 'final' / 'selected_params.toml')
        return selected

    def _fit_final_fusion_artifact(self, selected: dict[str, dict[str, Any]]) -> Path:
        return fit_final_fusion_artifact(
            self.model_cls,
            self.base_config,
            self.model_name,
            parameter_map=selected,
        )

    def _fit_final_models(
        self, selected: dict[str, dict[str, Any]], artifact: Path | None
    ) -> None:
        config = self._training_config({'train.final': True})
        if artifact is not None:
            publish_fusion_artifact(
                artifact,
                config.train._log_dir,
                list(config.dataset.names),
            )

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
