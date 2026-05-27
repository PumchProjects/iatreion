import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax

from iatreion.configs import DataName, DiscreteRrlConfig, ZeroMeanFallback
from iatreion.exceptions import IatreionException
from iatreion.train_utils import TrainStepContext
from iatreion.train_utils.fusion import (
    FUSION_ARTIFACT_FILE,
    AvailableFusionArtifact,
    ModalityCalibrator,
)
from iatreion.train_utils.imputation import (
    SimpleImputerArtifact,
    get_simple_imputer_path,
)
from iatreion.utils import decode_string

from .base import Model


@dataclass(frozen=True)
class RuleEval:
    truth: 'pd.Series[bool]'
    valid: 'pd.Series[bool]'
    coverage: 'pd.Series[float]'


@dataclass
class Item(ABC):
    name: str

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]': ...

    @abstractmethod
    def eval_with_coverage(self, data: pd.DataFrame) -> RuleEval: ...


def _eval_leaf(result: 'pd.Series[bool]', observed: 'pd.Series[bool]') -> RuleEval:
    truth = result.fillna(False).astype(bool) & observed
    return RuleEval(
        truth=truth,
        valid=observed,
        coverage=observed.astype(float),
    )


@dataclass
class BinaryItem(Item):
    @override
    def __str__(self) -> str:
        return self.name

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]':
        value = data[self.name]
        return value == 1

    @override
    def eval_with_coverage(self, data: pd.DataFrame) -> RuleEval:
        value = data[self.name]
        return _eval_leaf(value == 1, value.notna())


@dataclass
class DiscreteItem(Item):
    value: str | float | int
    category: str | None = None

    @override
    def __str__(self) -> str:
        return f'{self.name} = {self.category or self.value}'

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]':
        value = data[self.name]
        return value == self.value

    @override
    def eval_with_coverage(self, data: pd.DataFrame) -> RuleEval:
        value = data[self.name]
        return _eval_leaf(value == self.value, value.notna())


@dataclass
class ContinuousItem(Item):
    op: str
    th: float

    @override
    def __str__(self) -> str:
        match self.op:
            case '<=':
                op_str = '≤'
            case '>=':
                op_str = '≥'
            case _:
                op_str = self.op
        return f'{self.name} {op_str} {self.th:.3f}'

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]':
        value = data[self.name]
        match self.op:
            case '<':
                return value < self.th
            case '<=':
                return value <= self.th
            case '>':
                return value > self.th
            case '>=':
                return value >= self.th
            case _op:
                raise ValueError(f'Unknown operator: {_op}!')

    @override
    def eval_with_coverage(self, data: pd.DataFrame) -> RuleEval:
        value = data[self.name]
        return _eval_leaf(self.eval(data), value.notna())


def get_item(item: str) -> Item:
    item = item.strip()
    units = item.split()
    if len(units) == 1:
        units = units[0].split('_')
        if len(units) == 3:
            return DiscreteItem(units[0], int(units[1]), units[2])
        elif len(units) == 1:
            return BinaryItem(item)
        else:
            try:
                return DiscreteItem(units[0], float(units[1]))
            except ValueError:
                return DiscreteItem(*units)
    elif len(units) == 3:
        return ContinuousItem(units[0], units[1], float(units[2]))
    else:
        raise ValueError(f'Unit length != 1 or 3: {item}')


class Rule:
    def __init__(self, rule: str, *, is_not: bool = False) -> None:
        self.is_not = is_not
        self.op: str = '&'
        self.items: list[Rule | Item] = []
        counter, left, right = 0, 0, 0
        left_updated, right_updated = False, False
        for i, char in enumerate(rule):
            match char:
                case '(':
                    if counter == 0:
                        left = i + 1
                    counter += 1
                case ')':
                    counter -= 1
                    if counter == 0:
                        right_updated = True
                        right = i
                case '~':
                    if counter == 0:
                        left_updated = True
                        left = i + 1
                case '&' | '|':
                    if counter == 0:
                        self.op = char
                        if not right_updated:
                            right = i
                            self.items.append(
                                Rule(rule[left:right], is_not=True)
                                if left_updated
                                else get_item(rule[left:right])
                            )
                        else:
                            right_updated = False
                            self.items.append(
                                Rule(rule[left:right], is_not=left_updated)
                            )
                        left_updated = False
                        left = i + 1
        if not right_updated:
            right = len(rule)
            self.items.append(
                Rule(rule[left:right], is_not=True)
                if left_updated
                else get_item(rule[left:right])
            )
        else:
            self.items.append(Rule(rule[left:right], is_not=left_updated))

    def __str__(self) -> str:
        op_str = ' and ' if self.op == '&' else ' or '
        inner = op_str.join(str(item) for item in self.items)
        return f'{"not " if self.is_not else ""}({inner})'

    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]':
        result = self.items[0].eval(data)
        for item in self.items[1:]:
            other = item.eval(data)
            match self.op:
                case '|':
                    result |= other
                case '&':
                    result &= other
                case _op:
                    raise ValueError(f'Unknown operator: {_op}!')
        if self.is_not:
            result = ~result
        return result

    def eval_with_coverage(self, data: pd.DataFrame, *, tau: float) -> RuleEval:
        child_results = [
            item.eval_with_coverage(data, tau=tau)
            if isinstance(item, Rule)
            else item.eval_with_coverage(data)
            for item in self.items
        ]
        truth_frame = pd.DataFrame(
            {i: result.truth for i, result in enumerate(child_results)},
            index=data.index,
        )
        valid_frame = pd.DataFrame(
            {i: result.valid for i, result in enumerate(child_results)},
            index=data.index,
        )
        match self.op:
            case '|':
                truth = truth_frame.where(valid_frame, False).any(axis=1)
            case '&':
                truth = truth_frame.where(valid_frame, True).all(axis=1)
            case _op:
                raise ValueError(f'Unknown operator: {_op}!')
        if self.is_not:
            truth = ~truth
        coverage = valid_frame.mean(axis=1)
        valid = coverage >= tau
        return RuleEval(truth=truth, valid=valid, coverage=coverage)


@dataclass(frozen=True)
class RuleTableSchema:
    num_labels: int
    support_idx: int
    rule_idx: int
    mean_coverage_idx: int | None = None


class Line:
    def __init__(
        self,
        line: str,
        labels: list[str],
        callback: Callable[[str], str] | None,
        *,
        schema: RuleTableSchema,
        tau: float | None = None,
    ) -> None:
        units = line.rstrip().split('\t')
        self.rule_id = units[0]
        self.weights = list(map(float, units[1 : 1 + schema.num_labels]))
        self.support = float(units[schema.support_idx])
        self.mean_coverage = (
            float(units[schema.mean_coverage_idx])
            if schema.mean_coverage_idx is not None
            else None
        )
        self.tau = tau
        self.rule = Rule(units[schema.rule_idx])
        self.labels = labels
        self.callback = callback

    def print_rule(self) -> str:
        rule = decode_string(str(self.rule)[1:-1])
        if self.callback is not None:
            rule = self.callback(rule)
        return rule

    def activation(self, data: pd.DataFrame) -> RuleEval:
        if self.tau is None:
            truth = self.rule.eval(data).fillna(False).astype(bool)
            valid = pd.Series(True, index=data.index)
            return RuleEval(truth=truth, valid=valid, coverage=valid.astype(float))
        return self.rule.eval_with_coverage(data, tau=self.tau)

    def eval(
        self, data: pd.DataFrame, active_lines: list[Self] | None = None
    ) -> pd.DataFrame:
        result = self.activation(data)
        active = result.truth & result.valid
        if active_lines is not None and active.item():
            active_lines.append(self)
        active = active.astype(float)
        table = {
            label: active * weight
            for label, weight in zip(self.labels, self.weights, strict=True)
        }
        return pd.DataFrame(table, dtype=float)


class Rrl:
    metadata_template = re.compile(
        r"""
            (?:RID|Meta) \(               # legacy RID(...) or new Meta(...)
            et = (?P<train_err> .*? ) ,    # train error
            ft = (?P<train_f1> .*? ) ,     # train f1
            ev = (?P<val_err> .*? ) ,      # val error
            fv = (?P<val_f1> .*? ) ,       # val f1
            t = (?P<temp> .*? )            # temperature
            (?: , tau = (?P<tau> .*? ) )?  # optional coverage tau
            \)                             # )
        """,
        re.VERBOSE,
    )
    label_template = re.compile(r'(?P<label>.*)\(b=(?P<bias>.*)\)')

    def __init__(
        self, file: Path, callback: Callable[[str], str] | None = None
    ) -> None:
        with file.open('r', encoding='utf-8') as f:
            texts = [line.rstrip('\n') for line in f if line.strip()]

        metadata, headers, rule_lines = self._split_metadata_and_table(texts)
        match_obj = (
            None if metadata is None else self.metadata_template.fullmatch(metadata)
        )
        self.weight = 1.0
        if match_obj is not None:
            self.temp = float(match_obj.group('temp'))
            tau = match_obj.group('tau')
            self.tau = None if tau is None else float(tau)
        else:
            self.temp = 0.01
            self.tau = None
        self.imputer: SimpleImputerArtifact | None = (
            None
            if self.tau is not None
            else SimpleImputerArtifact.load(get_simple_imputer_path(file))
        )

        self.labels, self.biases, schema = self._parse_table_header(headers)
        self.lines = [
            Line(line, self.labels, callback, schema=schema, tau=self.tau)
            for line in rule_lines
        ]

    @classmethod
    def _split_metadata_and_table(
        cls, texts: list[str]
    ) -> tuple[str | None, list[str], list[str]]:
        first_units = texts[0].split('\t')
        if len(first_units) == 1 and cls.metadata_template.fullmatch(first_units[0]):
            return first_units[0], texts[1].split('\t'), texts[2:]

        metadata = (
            first_units[0] if cls.metadata_template.fullmatch(first_units[0]) else None
        )
        return metadata, first_units, texts[1:]

    @classmethod
    def _parse_table_header(
        cls, headers: list[str]
    ) -> tuple[list[str], list[float], RuleTableSchema]:
        labels: list[str] = []
        biases: list[float] = []
        column_start = 1
        for header in headers[column_start:]:
            match_obj = cls.label_template.fullmatch(header)
            if match_obj is None:
                break
            labels.append(match_obj.group('label').split('_')[-1])
            biases.append(float(match_obj.group('bias')))

        named_columns = {name: idx for idx, name in enumerate(headers)}
        schema = RuleTableSchema(
            num_labels=len(labels),
            support_idx=named_columns['Support'],
            mean_coverage_idx=named_columns.get('MeanCoverage'),
            rule_idx=named_columns['Rule'],
        )
        return labels, biases, schema

    def _make_empty_result(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {label: [0.0] * len(data) for label in self.labels},
            dtype='Float64',
            index=data.index,
        )

    def _add_bias(self, result: pd.DataFrame) -> None:
        for label, bias in zip(self.labels, self.biases, strict=True):
            result[label] += bias

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None:
            return data
        return self.imputer.apply(data)

    def _iter_enabled_lines(self, enabled_rule_indices: list[int] | None) -> list[Line]:
        if enabled_rule_indices is None:
            return self.lines

        indices = set(enabled_rule_indices)
        invalid = sorted(idx for idx in indices if idx < 0 or idx >= len(self.lines))
        if invalid:
            raise IatreionException(
                'Invalid RRL rule indices: $indices. Available range: 0-$last.',
                indices=', '.join(map(str, invalid)),
                last=str(len(self.lines) - 1),
            )
        return [line for i, line in enumerate(self.lines) if i in indices]

    def _apply_zero_mean_fallback(
        self,
        score: pd.DataFrame,
        zero_mean_fallback: ZeroMeanFallback,
    ) -> None:
        if zero_mean_fallback == 'uniform':
            return

        zero_rows = pd.Series(
            np.isclose(score.to_numpy(dtype=float), 0.0).all(axis=1),
            index=score.index,
        )
        if zero_rows.any():
            bias_label = self.labels[np.argmax(self.biases).item()]
            score.loc[zero_rows, bias_label] += 1e-3

    def score(
        self,
        data: pd.DataFrame,
        active_lines: list[Line] | None = None,
        *,
        bias_enabled: bool = True,
        enabled_rule_indices: list[int] | None = None,
        zero_mean_fallback: ZeroMeanFallback = 'uniform',
    ) -> pd.DataFrame:
        result = self._make_empty_result(data)
        if bias_enabled:
            self._add_bias(result)
        for line in self._iter_enabled_lines(enabled_rule_indices):
            result += line.eval(data, active_lines)
        result = result.astype(float)
        self._apply_zero_mean_fallback(result, zero_mean_fallback)
        return result

    def eval(
        self,
        data: pd.DataFrame,
        active_lines: list[Line] | None = None,
        *,
        bias_enabled: bool = True,
        enabled_rule_indices: list[int] | None = None,
        zero_mean_fallback: ZeroMeanFallback = 'uniform',
    ) -> pd.DataFrame:
        score = self.score(
            data,
            active_lines,
            bias_enabled=bias_enabled,
            enabled_rule_indices=enabled_rule_indices,
            zero_mean_fallback=zero_mean_fallback,
        )
        # Returned results all have "float64" dtype
        return self.softmax(score)

    @staticmethod
    def softmax(score: pd.DataFrame) -> pd.DataFrame:
        return score.apply(softmax, axis=1, raw=True, result_type='expand')


def transform_rrl_score_space(
    rrl: Rrl,
    *,
    alpha: float,
    calibrator: ModalityCalibrator,
    positive_label: str,
) -> Rrl:
    transformed = deepcopy(rrl)
    scale = alpha * calibrator.slope
    transformed.biases = [scale * bias for bias in transformed.biases]
    for line in transformed.lines:
        line.weights = [scale * weight for weight in line.weights]

    try:
        positive_index = transformed.labels.index(positive_label)
    except ValueError as exc:
        raise IatreionException(
            'RRL labels [$labels] do not contain positive fusion label "$label".',
            labels=', '.join(transformed.labels),
            label=positive_label,
        ) from exc
    transformed.biases[positive_index] += alpha * calibrator.intercept
    transformed.weight = alpha
    return transformed


class DiscreteRrlModel(Model):
    def __init__(
        self,
        config: DiscreteRrlConfig,
        callbacks: list[Callable[[str], str] | None] | None = None,
    ) -> None:
        super().__init__()
        self.config: DiscreteRrlConfig = config
        self.callbacks: list[Callable[[str], str] | None] = (
            callbacks
            if callbacks is not None
            else [None for _ in range(len(config.dataset.names))]
        )
        self.ctx: TrainStepContext | None = None
        self._artifact: AvailableFusionArtifact | None = None

    @property
    def artifact(self) -> AvailableFusionArtifact:
        if self._artifact is None:
            self._artifact = AvailableFusionArtifact.load(
                self.config.rrl_root / FUSION_ARTIFACT_FILE
            )
        return self._artifact

    def _validate_artifact_names(self, names: list[DataName]) -> None:
        missing = sorted(set(names) - set(self.artifact.names))
        if not missing:
            return
        raise IatreionException(
            'Available-fusion artifact does not contain module(s): $names.',
            names=', '.join(missing),
        )

    def get_model(self, ctx: TrainStepContext) -> Rrl:
        return Rrl(self.config.rrl_root / ctx.rrl_file)

    def get_raw_models(self) -> list[Rrl]:
        # HACK: Coupled with TrainStepContext.rrl_file
        # TODO: Unimplemented when TrainConfig.aggregate is 'concat'
        return [
            Rrl(self.config.rrl_root / f'{name}.tsv', callback)
            for name, callback in zip(
                self.config.dataset.names, self.callbacks, strict=True
            )
        ]

    def get_models(self) -> list[Rrl]:
        names = self.config.dataset.names
        self._validate_artifact_names(names)
        return [
            transform_rrl_score_space(
                rrl,
                alpha=self.artifact.weights[name],
                calibrator=self.artifact.calibrators[name],
                positive_label=self.artifact.positive_label,
            )
            for name, rrl in zip(names, self.get_raw_models(), strict=True)
        ]

    @override
    def _fit(self, X: NDArray, y: NDArray) -> None:
        raise NotImplementedError

    @override
    def fit(self, ctx: TrainStepContext) -> None:
        self.ctx = ctx

    @override
    def _predict_proba(self, X: NDArray) -> NDArray:
        assert self.ctx is not None
        data = pd.DataFrame(X, columns=self.ctx.db_enc.X_fname)
        result = self.get_model(self.ctx).eval(data)
        return result.to_numpy()

    def _validate_enabled_terms(
        self,
        names: list[DataName],
        models: list[Rrl],
        enabled_biases: dict[str, bool] | None,
        enabled_rules: dict[str, list[int]] | None,
    ) -> None:
        available_names = set(names)
        selected_names = set(enabled_biases or {}) | set(enabled_rules or {})
        invalid_names = sorted(selected_names - available_names)
        if invalid_names:
            raise IatreionException(
                'Unknown RRL module selection "$name". Available modules: $available.',
                name=', '.join(invalid_names),
                available=', '.join(names),
            )

        if enabled_rules is None:
            return
        model_map = dict(zip(names, models, strict=True))
        for name, indices in enabled_rules.items():
            line_count = len(model_map[name].lines)
            invalid_indices = sorted(
                {idx for idx in indices if idx < 0 or idx >= line_count}
            )
            if invalid_indices:
                raise IatreionException(
                    'Invalid RRL rule indices for "$name": $indices. '
                    'Available range: 0-$last.',
                    name=name,
                    indices=', '.join(map(str, invalid_indices)),
                    last=str(line_count - 1),
                )

    @staticmethod
    def _combined_index(data: list[pd.DataFrame]) -> pd.Index:
        index = data[0].index
        for frame in data[1:]:
            index = index.union(frame.index, sort=False)
        return index

    @staticmethod
    def _available_mask(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
        available = ~frame.isna().all(axis=1)
        return available.reindex(index, fill_value=False).astype(bool)

    def eval(
        self,
        data: list[pd.DataFrame],
        *,
        enabled_biases: dict[str, bool] | None = None,
        enabled_rules: dict[str, list[int]] | None = None,
        zero_mean_fallback: ZeroMeanFallback = 'uniform',
    ) -> pd.DataFrame:
        names = self.config.dataset.names
        raw_models = self.get_raw_models()
        self._validate_artifact_names(names)
        self._validate_enabled_terms(names, raw_models, enabled_biases, enabled_rules)

        index = self._combined_index(data)
        y_pos_score_list: list[NDArray] = []
        y_mask_list: list[NDArray] = []
        for name, X, model in zip(names, data, raw_models, strict=True):
            result = model.eval(
                model.impute(X),
                bias_enabled=(
                    True if enabled_biases is None else enabled_biases.get(name, True)
                ),
                enabled_rule_indices=(
                    None if enabled_rules is None else enabled_rules.get(name)
                ),
                zero_mean_fallback=zero_mean_fallback,
            )
            available = self._available_mask(X, index)
            y_mask_list.append((~available).to_numpy(dtype=bool))
            y_pos_score = result.reindex(index)[self.artifact.positive_label]
            y_pos_score_list.append(y_pos_score.fillna(0.5).to_numpy())

        result = pd.DataFrame(
            self.artifact.predict_scores(names, y_pos_score_list, y_mask_list),
            index=index,
            columns=self.artifact.labels,
        )
        available_any = ~np.column_stack(y_mask_list).all(axis=1)
        result.loc[~available_any] = np.nan
        return result

    def predict_labels(self, result: pd.DataFrame) -> pd.Series:
        y_pos_score = result[self.artifact.positive_label].to_numpy()
        labels = self.artifact.predict_labels(y_pos_score)
        labels[pd.isna(result).all(axis=1).to_numpy()] = ''
        return pd.Series(labels, index=result.index, name='Label')

    def interpret(
        self, data: list[pd.DataFrame]
    ) -> tuple[
        list[DataName],
        list[Rrl],
        list[pd.DataFrame],
        list[pd.DataFrame],
        list[tuple[DataName, Line]],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        names = self.config.dataset.names
        self._validate_artifact_names(names)
        raw_models = self.get_raw_models()

        available_names: list[DataName] = []
        available_data: list[pd.DataFrame] = []
        available_models: list[Rrl] = []
        for name, X, model in zip(names, data, raw_models, strict=True):
            if len(X) != 1:
                raise IatreionException(
                    'RRL interpretation requires exactly one sample '
                    'for "$name"; got $n.',
                    name=name,
                    n=str(len(X)),
                )
            if X.isna().all(axis=1).item():
                continue
            available_names.append(name)
            available_data.append(model.impute(X))
            available_models.append(model)

        if not available_names:
            raise IatreionException('No available RRL modules for interpretation.')

        normalized_weights = self.artifact.normalized_weights(available_names)
        models = [
            transform_rrl_score_space(
                model,
                alpha=normalized_weights[name],
                calibrator=self.artifact.calibrators[name],
                positive_label=self.artifact.positive_label,
            )
            for name, model in zip(available_names, available_models, strict=True)
        ]

        predictions: list[pd.DataFrame] = []
        active_lines: list[tuple[DataName, Line]] = []
        score_parts: list[pd.DataFrame] = []
        for name, X, model in zip(available_names, available_data, models, strict=True):
            lines: list[Line] = []
            score = model.score(X, lines)
            predictions.append(model.softmax(score))
            active_lines += [(name, line) for line in lines]
            score_parts.append(score)

        final_score = score_parts[0].copy()
        for score in score_parts[1:]:
            final_score += score
        result = Rrl.softmax(final_score)
        return (
            available_names,
            models,
            score_parts,
            predictions,
            active_lines,
            final_score,
            result,
        )
