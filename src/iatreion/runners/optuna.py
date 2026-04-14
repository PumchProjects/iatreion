from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, override

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.study import Study
from optuna.trial import Trial

from iatreion.configs import ModelConfig
from iatreion.models import Model
from iatreion.trainers import ModelTrainer
from iatreion.utils import load_dict, logger, save_dict

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
        kind = data['type']
        return cls(
            kind=kind,
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
                    name,
                    float(self.low),
                    float(self.high),
                    log=self.log,
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
    storage: str | None = None
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
            name=data['name'],
            objective=data['objective'],
            direction=data.get('direction', 'maximize'),
            storage=data.get('storage'),
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'TuningExecutionConfig':
        return cls(
            trial_log_root=Path(data.get('trial_log_root', 'logs_optuna')),
            fail_value=float(data.get('fail_value', 0.0)),
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
        execution = TuningExecutionConfig.from_dict(data.get('execution', {}))
        if config.study_name is not None:
            data['study']['name'] = config.study_name
        assert (name := data['study'].get('name')), 'Study name is required'
        storage_path = execution.trial_log_root / name / 'study.db'
        data['study']['storage'] = f'sqlite:///{storage_path}'
        return cls(
            study=TuningStudyConfig.from_dict(data['study']),
            execution=execution,
            search=flatten_search_space(data['search']),
        )

    @property
    def study_root(self) -> Path:
        return self.execution.trial_log_root / self.study.name


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


def apply_overrides(obj: Any, overrides: dict[str, Any]) -> Any:
    direct: dict[str, Any] = {}
    nested: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in overrides.items():
        head, _, tail = key.partition('.')
        if not tail:
            direct[head] = value
            continue
        nested[head][tail] = value

    for head, child_overrides in nested.items():
        direct[head] = apply_overrides(getattr(obj, head), child_overrides)
    return replace(obj, **direct)


class OptunaRunner(Runner):
    def __init__(self, model_cls: type[Model], config: ModelConfig) -> None:
        super().__init__(model_cls, config)
        self.spec = TuningSpec.load(config)

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

    def _create_study(self) -> Study:
        self.spec.study_root.mkdir(parents=True, exist_ok=True)
        return optuna.create_study(
            study_name=self.spec.study.name,
            storage=self.spec.study.storage,
            direction=self.spec.study.direction,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
            load_if_exists=self.spec.study.load_if_exists,
        )

    def _build_trial_config(self, trial: Trial) -> tuple[ModelConfig, dict[str, Any]]:
        sampled: dict[str, Any] = {}
        overrides: dict[str, Any] = {}
        for key, space in self.spec.search.items():
            value = space.sample(trial, key)
            sampled[key] = value
            overrides[key] = value

        trial_log_root = self.spec.study_root / f'trial_{trial.number:04d}'
        overrides['train.log_root'] = trial_log_root
        overrides['tune_config'] = None
        config = apply_overrides(self.base_config, overrides)
        return config, sampled

    def _run_trial(self, trial: Trial) -> float:
        config, sampled = self._build_trial_config(trial)
        model: Model | None = None
        try:
            model = self.model_cls(config)
            trainer = ModelTrainer(model)
            summary = trainer.train()
        except Exception as error:
            trial.set_user_attr('status', 'failed')
            trial.set_user_attr('error', repr(error))
            logger.exception(f'Optuna trial {trial.number} failed')
            return self.spec.execution.fail_value
        finally:
            if model is not None:
                model.close()
            config.close_log_handler()

        objective_name = self.spec.study.objective
        objective = summary.objectives.get(objective_name)
        if objective is None:
            trial.set_user_attr('status', 'missing-objective')
            trial.set_user_attr('objective_name', objective_name)
            return self.spec.execution.fail_value

        trial.set_user_attr('status', 'completed')
        trial.set_user_attr('log_root', str(config.train.log_root))
        trial.set_user_attr('sampled_params', sampled)
        for key, value in summary.objectives.items():
            trial.set_user_attr(key, float(value))
        return float(objective)

    @override
    def run(self) -> None:
        study = self._create_study()
        logger.info(
            'Optuna study "%s" started with objective %s',
            self.spec.study.name,
            self.spec.study.objective,
        )
        study.optimize(
            self._run_trial,
            n_trials=self.spec.study.n_trials,
            timeout=self.spec.study.timeout_sec,
        )
        if study.trials:
            logger.info(
                'Best trial #%s: value=%.6f params=%s',
                study.best_trial.number,
                study.best_value,
                study.best_trial.params,
            )
            info = {
                'trial_number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_trial.params,
                'user_attrs': study.best_trial.user_attrs,
            }
            save_dict(info, self.spec.study_root / 'best_trial.toml')
