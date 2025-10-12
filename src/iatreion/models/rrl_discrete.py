import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax

from iatreion.configs import DiscreteRrlConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import logger

from .base import ModelReturn, RawModel


@dataclass
class Item(ABC):
    name: str

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def eval(self, data: pd.DataFrame) -> 'pd.Series[pd.BooleanDtype]': ...


@dataclass
class BinaryItem(Item):
    @override
    def __str__(self) -> str:
        return self.name

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[pd.BooleanDtype]':
        value = data[self.name]
        return value == 1


@dataclass
class DiscreteItem(Item):
    value: str | float

    @override
    def __str__(self) -> str:
        return f'{self.name} = {self.value}'

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[pd.BooleanDtype]':
        value = data[self.name]
        return value == self.value


@dataclass
class ContinuousItem(Item):
    op: str
    th: float

    @override
    def __str__(self) -> str:
        return f'{self.name} {self.op} {self.th}'

    @override
    def eval(self, data: pd.DataFrame) -> 'pd.Series[pd.BooleanDtype]':
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
        try:
            return ContinuousItem(units[0], units[1], float(units[2]))
        except ValueError:
            # HACK: Make sure that units[2] cannot be converted to float
            return BinaryItem(item)
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

    def eval(self, data: pd.DataFrame) -> 'pd.Series[pd.BooleanDtype]':
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

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.rule.eval(data)
        return pd.DataFrame(
            {
                label: result * weight
                for label, weight in zip(self.labels, self.weights, strict=False)
            }
        )

    def interpret(self, data: pd.DataFrame, active_lines: list[Self]) -> pd.DataFrame:
        result = self.rule.eval(data)
        if not pd.isna(r := result.item()) and r:
            active_lines.append(self)
        return pd.DataFrame(
            {
                label: result * weight
                for label, weight in zip(self.labels, self.weights, strict=False)
            }
        )


class Rrl:
    rid_template = re.compile(r'RID\(t=(?P<temp>.*)\)')
    label_template = re.compile(r'(?P<label>.*)\(b=(?P<bias>.*)\)')

    def __init__(self, file: Path) -> None:
        with file.open('r', encoding='utf-8') as f:
            texts = f.readlines()
        headers = texts[0].split('\t')
        self.labels: list[str] = []
        self.biases: list[float] = []
        match_obj = self.rid_template.match(headers[0])
        if match_obj is not None:
            self.temp = float(match_obj.group('temp'))
        else:
            self.temp = 0.01
            logger.warning(
                f'[bold yellow]Using default temperature {self.temp} for old versions',
                extra={'markup': True},
            )
        for header in headers[1:-2]:
            match_obj = self.label_template.match(header)
            assert match_obj is not None, f'Invalid header: {header}!'
            self.labels.append(match_obj.group('label').split('_')[-1])
            self.biases.append(float(match_obj.group('bias')))
        self.lines = [Line(line, self.labels) for line in texts[1:]]

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                label: [bias] * len(data)
                for label, bias in zip(self.labels, self.biases, strict=False)
            },
            dtype='Float64',
            index=data.index,
        )
        for line in self.lines:
            result += line.eval(data)
        return result

    def predict(self, data: pd.DataFrame) -> NDArray:
        result = self.eval(data).astype(float).values
        return softmax(result / self.temp, axis=1)

    def interpret(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[Line]]:
        result = pd.DataFrame(
            {
                label: [bias]
                for label, bias in zip(self.labels, self.biases, strict=False)
            },
            dtype='Float64',
            index=data.index,
        )
        active_lines: list[Line] = []
        for line in self.lines:
            result += line.interpret(data, active_lines)
        return result, active_lines


class DiscreteRrlModel(RawModel):
    def __init__(self, config: DiscreteRrlConfig) -> None:
        super().__init__()
        self.config = config
        exp_root = self.config.get_best_exp_root()
        if exp_root is None:
            raise IatreionException(
                'No experiment root found for $dataset and groups "$groups".',
                dataset=self.config.dataset.name,
                groups=self.config.train.group_names,
            )
        self.exp_root = exp_root

    def get_rrl(self) -> Rrl:
        return Rrl(self.config.get_rrl_file(self.exp_root))

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        predicted = self.get_rrl().predict(X)
        return predicted, {}

    def eval(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.get_rrl().eval(data)

    def interpret(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[Line], Rrl]:
        rrl = self.get_rrl()
        result, active_lines = rrl.interpret(data.head(1))
        return result, active_lines, rrl
