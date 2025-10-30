import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast, override

import pandas as pd
from scipy.special import expit, softmax

from iatreion.configs import DataName, DiscreteRrlConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import decode_string, logger

from .base import ModelReturn, RawModel


@dataclass
class Item(ABC):
    name: str

    @property
    def true_name(self) -> str:
        return decode_string(self.name)

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]': ...


@dataclass
class BinaryItem(Item):
    @override
    def __str__(self) -> str:
        return self.true_name

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]':
        value = data[self.name]
        return value == 1


@dataclass
class DiscreteItem(Item):
    value: str | float

    @override
    def __str__(self) -> str:
        return f'{self.true_name} = {self.value}'

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[bool]':
        value = data[self.name]
        return value == self.value


@dataclass
class ContinuousItem(Item):
    op: str
    th: float

    @override
    def __str__(self) -> str:
        return f'{self.true_name} {self.op} {self.th:.3f}'

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


def get_item(item: str) -> Item:
    item = item.strip()
    units = item.split()
    if len(units) == 1:
        units = units[0].split('_')
        if len(units) == 1:
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
    def __init__(self, rule: str, is_not: bool = False) -> None:
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
                                Rule(rule[left:right], True)
                                if left_updated
                                else get_item(rule[left:right])
                            )
                        else:
                            right_updated = False
                            self.items.append(Rule(rule[left:right], left_updated))
                        left_updated = False
                        left = i + 1
        if not right_updated:
            right = len(rule)
            self.items.append(
                Rule(rule[left:right], True)
                if left_updated
                else get_item(rule[left:right])
            )
        else:
            self.items.append(Rule(rule[left:right], left_updated))

    def __str__(self) -> str:
        inner = f' {self.op} '.join([str(item) for item in self.items])
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
        if self.is_not:
            result = ~result
        return result


class Line:
    def __init__(self, line: str, labels: list[str]) -> None:
        units = line.split('\t')
        self.weights = list(map(float, units[1:-2]))
        self.support = float(units[-2])
        self.rule = Rule(units[-1])
        self.labels = labels

    def print_rule(self) -> str:
        return str(self.rule)[1:-1]

    def eval(
        self, data: pd.DataFrame, active_lines: list[Self] | None = None
    ) -> pd.DataFrame:
        result = self.rule.eval(data)
        if active_lines is not None and not pd.isna(r := result.item()) and r:
            active_lines.append(self)
        table: dict[str, pd.Series] = {}
        for label, weight in zip(self.labels, self.weights, strict=False):
            col = result * weight
            table[f'{label}_upper'] = col.fillna(max(0, weight))
            table[f'{label}_lower'] = col.fillna(min(0, weight))
        return pd.DataFrame(table)


class Rrl:
    rid_template = re.compile(r'RID\(w=(?P<weight>.*),t=(?P<temp>.*)\)')
    label_template = re.compile(r'(?P<label>.*)\(b=(?P<bias>.*)\)')

    def __init__(self, file: Path) -> None:
        with file.open('r', encoding='utf-8') as f:
            texts = f.readlines()
        headers = texts[0].split('\t')

        match_obj = self.rid_template.match(headers[0])
        if match_obj is not None:
            self.temp = float(match_obj.group('temp'))
            self.weight = float(match_obj.group('weight'))
        else:
            self.temp = 0.01
            self.weight = 1.0
            logger.warning(
                f'[bold yellow]Using default temperature {self.temp}'
                f' and weight {self.weight} for old versions',
                extra={'markup': True},
            )

        self.labels: list[str] = []
        self.biases: list[float] = []
        for header in headers[1:-2]:
            match_obj = self.label_template.match(header)
            assert match_obj is not None, f'Invalid header: {header}!'
            self.labels.append(match_obj.group('label').split('_')[-1])
            self.biases.append(float(match_obj.group('bias')))

        self.lines = [Line(line, self.labels) for line in texts[1:]]

    def eval(
        self, data: pd.DataFrame, active_lines: list[Line] | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        result = pd.DataFrame(
            {
                name: [bias] * len(data)
                for label, bias in zip(self.labels, self.biases, strict=False)
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
            }
        )
        max_lower = result[[f'{label}_lower' for label in self.labels]].max(axis=1)
        min_upper = result[[f'{label}_upper' for label in self.labels]].min(axis=1)
        confidence = (max_lower - min_upper).apply(expit)
        return mean_result.apply(softmax, axis=1, result_type='expand'), confidence


class DiscreteRrlModel(RawModel):
    def __init__(self, config: DiscreteRrlConfig) -> None:
        super().__init__()
        self.config = config
        self.models = [
            Rrl(self.config.get_rrl_file(exp_root))
            for exp_root in config.get_exp_roots()
        ]

    def aggregate(
        self, predictions: list[tuple[pd.DataFrame, pd.Series]]
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not predictions:
            raise IatreionException('No predictions to aggregate!')
        dividends: list[pd.DataFrame] = []
        divisors: list[pd.Series] = []
        for (pred, confidence), model in zip(predictions, self.models, strict=False):
            composite_weight = confidence * model.weight
            dividends.append(pred.mul(composite_weight, axis=0))
            divisors.append(composite_weight)
        dividend = cast(pd.DataFrame, sum(dividends))
        divisor = cast(pd.Series, sum(divisors))
        results = dividend.div(divisor + 1e-8, axis=0)
        confidence = pd.concat([c for _, c in predictions], axis=1).max(axis=1)
        results.loc[confidence < 0.5] = pd.NA
        return results, confidence

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        predictions = [model.eval(X) for model in self.models]
        results, _ = self.aggregate(predictions)
        return results.to_numpy('float32'), {}

    def eval(self, data: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
        predictions = [
            model.eval(X) for X, model in zip(data, self.models, strict=False)
        ]
        results, confidence = self.aggregate(predictions)
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
        predictions: list[tuple[pd.DataFrame, pd.Series]] = []
        active_lines: list[tuple[DataName, Line]] = []
        for name, X, model in zip(names, data, self.models, strict=False):
            lines: list[Line] = []
            predictions.append(model.eval(X.head(1), lines))
            active_lines += [(name, line) for line in lines]
        result, confidence = self.aggregate(predictions)
        return names, self.models, predictions, active_lines, result, confidence
