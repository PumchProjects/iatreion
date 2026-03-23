import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit, softmax

from iatreion.configs import DataName, DiscreteRrlConfig
from iatreion.exceptions import IatreionException
from iatreion.train_utils import TrainStepContext
from iatreion.utils import decode_string, logger

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
        return f'{self.name} {self.op} {self.th:.3f}'

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
        inner = f' {self.op} '.join(str(item) for item in self.items)
        return f'{"~" if self.is_not else ""}({inner})'

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

    def eval(
        self, data: pd.DataFrame, active_lines: list[Self] | None = None
    ) -> pd.DataFrame:
        if self.tau is None:
            result = self.rule.eval(data)
            if active_lines is not None and not pd.isna(r := result.item()) and r:
                active_lines.append(self)
            table: dict[str, pd.Series] = {}
            for label, weight in zip(self.labels, self.weights, strict=True):
                col = result * weight
                table[f'{label}_upper'] = col.fillna(max(0, weight))
                table[f'{label}_lower'] = col.fillna(min(0, weight))
            return pd.DataFrame(table)

        gated = self.rule.eval_with_coverage(data, tau=self.tau)
        result = gated.truth & gated.valid
        if active_lines is not None and result.item():
            active_lines.append(self)
        active = result.astype(float)
        table = {
            name: active * weight
            for label, weight in zip(self.labels, self.weights, strict=True)
            for name in (f'{label}_upper', f'{label}_lower')
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
        self, file: Path, weight: str, callback: Callable[[str], str] | None = None
    ) -> None:
        with file.open('r', encoding='utf-8') as f:
            texts = [line.rstrip('\n') for line in f if line.strip()]

        metadata, headers, rule_lines = self._split_metadata_and_table(texts)
        match_obj = (
            None if metadata is None else self.metadata_template.fullmatch(metadata)
        )
        if match_obj is not None:
            self.temp = float(match_obj.group('temp'))
            tau = match_obj.group('tau')
            self.tau = None if tau is None else float(tau)
            match weight:
                case 'uniform':
                    self.weight = 1.0
                case 'train-f1':
                    self.weight = float(match_obj.group('train_f1'))
                case 'val-f1':
                    self.weight = float(match_obj.group('val_f1'))
                case 'train-adaboost':
                    error = float(match_obj.group('train_err'))
                    self.weight = 0.5 * np.log((1 - error) / (error + 1e-8))
                case 'val-adaboost':
                    error = float(match_obj.group('val_err'))
                    self.weight = 0.5 * np.log((1 - error) / (error + 1e-8))
                case _:
                    raise ValueError(f'Unknown weight mode: {weight}!')
        else:
            self.temp = 0.01
            self.tau = None
            self.weight = 1.0
            logger.warning(
                f'[bold yellow]Using default temperature {self.temp}'
                f' and weight {self.weight} for old versions',
                extra={'markup': True},
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

    def eval(
        self, data: pd.DataFrame, active_lines: list[Line] | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        result = pd.DataFrame(
            {
                name: [bias] * len(data)
                for label, bias in zip(self.labels, self.biases, strict=True)
                for name in (f'{label}_upper', f'{label}_lower')
            },
            dtype='Float64',
            index=data.index,
        )
        for line in self.lines:
            result += line.eval(data, active_lines)
        mean_result = pd.DataFrame(
            {
                label: (result[f'{label}_upper'] + result[f'{label}_lower']) / 2
                for label in self.labels
            },
            dtype=float,
        )
        softmax_result = mean_result.apply(
            softmax, axis=1, raw=True, result_type='expand'
        )
        if self.tau is None:
            max_lower = result[[f'{label}_lower' for label in self.labels]].max(axis=1)
            min_upper = result[[f'{label}_upper' for label in self.labels]].min(axis=1)
            confidence: pd.Series = (max_lower - min_upper).map(expit)
        else:
            confidence = pd.Series(np.nan, index=data.index, dtype=float)
        # Returned results all have "float64" dtype
        return softmax_result, confidence


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

    def get_model(self, ctx: TrainStepContext) -> Rrl:
        return Rrl(self.config.rrl_root / ctx.rrl_file, self.config._weight)

    def get_models(self) -> list[Rrl]:
        # HACK: Coupled with TrainStepContext.rrl_file
        # TODO: Unimplemented when TrainConfig.aggregate is 'concat'
        return [
            Rrl(self.config.rrl_root / f'{name}.tsv', self.config._weight, callback)
            for name, callback in zip(
                self.config.dataset.names, self.callbacks, strict=True
            )
        ]

    def aggregate(
        self, models: list[Rrl], predictions: list[tuple[pd.DataFrame, pd.Series]]
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not predictions:
            raise IatreionException('No predictions to aggregate!')

        results = predictions[0][0].copy()
        results.iloc[:, :] = 0.0
        confidence_parts: list[pd.Series] = []
        for (pred, confidence), model in zip(predictions, models, strict=True):
            if confidence.notna().any():
                pred_weight = confidence.fillna(1.0) * model.weight
                confidence_parts.append(confidence)
            else:
                pred_weight = pd.Series(model.weight, index=pred.index, dtype=float)
            results += pred.mul(pred_weight, axis=0)

        results = results.div(results.sum(axis=1) + 1e-8, axis=0)
        if confidence_parts:
            confidence = pd.concat(confidence_parts, axis=1).max(axis=1)
            results.loc[confidence < 0.5] = np.nan
        else:
            confidence = pd.Series(np.nan, index=results.index, dtype=float)
        return results, confidence

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
        result, _ = self.get_model(self.ctx).eval(data)
        return result.to_numpy()

    def eval(self, data: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
        models = self.get_models()
        predictions = [model.eval(X) for X, model in zip(data, models, strict=True)]
        results, confidence = self.aggregate(models, predictions)
        return results, confidence

    def interpret(
        self, data: list[pd.DataFrame]
    ) -> tuple[
        list[DataName],
        list[Rrl],
        list[tuple[pd.DataFrame, pd.Series]],
        list[tuple[DataName, Line]],
        pd.DataFrame,
        pd.Series,
    ]:
        names = self.config.dataset.names
        models = self.get_models()
        predictions: list[tuple[pd.DataFrame, pd.Series]] = []
        active_lines: list[tuple[DataName, Line]] = []
        for name, X, model in zip(names, data, models, strict=True):
            lines: list[Line] = []
            predictions.append(model.eval(X.head(1), lines))
            active_lines += [(name, line) for line in lines]
        result, confidence = self.aggregate(models, predictions)
        return names, models, predictions, active_lines, result, confidence
