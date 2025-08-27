import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax

from iatreion.configs import DiscreteRrlConfig
from iatreion.utils import logger

from .base import ModelReturn, RawModel


@dataclass
class Item(ABC):
    name: str

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def eval(self, data: pd.DataFrame) -> NDArray[np.bool_]: ...


@dataclass
class DiscreteItem(Item):
    value: str | float

    @override
    def __str__(self) -> str:
        return f'{self.name} = {self.value}'

    @override
    def eval(self, data: pd.DataFrame) -> NDArray[np.bool_]:
        value = data[self.name].to_numpy()
        return value == self.value


@dataclass
class ContinuousItem(Item):
    op: str
    th: float

    @override
    def __str__(self) -> str:
        return f'{self.name} {self.op} {self.th}'

    @override
    def eval(self, data: pd.DataFrame) -> NDArray[np.bool_]:
        value = data[self.name].to_numpy()
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
    units = item.strip().split()
    if len(units) == 1:
        units = units[0].split('_')
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

    def eval(self, data: pd.DataFrame) -> NDArray[np.bool_]:
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
    def __init__(self, line: str) -> None:
        units = line.split('\t')
        self.weights = list(map(float, units[1:-2]))
        self.support = float(units[-2])
        self.rule = Rule(units[-1])

    def print_rule(self) -> str:
        return str(self.rule)[1:-1]

    def eval(self, data: pd.DataFrame) -> NDArray:
        result = self.rule.eval(data)
        return np.stack([result * weight for weight in self.weights], axis=-1)

    def interpret(self, data: pd.DataFrame, active_lines: list[Self]) -> NDArray:
        result = self.rule.eval(data)
        if result.item():
            active_lines.append(self)
        return np.stack([result * weight for weight in self.weights], axis=-1)


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
        self.lines = [Line(line) for line in texts[1:]]

    def eval(self, data: pd.DataFrame, prob: bool = False) -> NDArray:
        result = np.repeat([self.biases], data.shape[0], axis=0)
        for line in self.lines:
            result += line.eval(data)
        if prob:
            return softmax(result / self.temp, axis=1)
        return result

    def interpret(self, data: pd.DataFrame) -> tuple[list[float], list[Line]]:
        result = np.repeat([self.biases], data.shape[0], axis=0)
        active_lines: list[Line] = []
        for line in self.lines:
            result += line.interpret(data, active_lines)
        result_list = result.squeeze().tolist()
        return result_list, active_lines


class DiscreteRrlModel(RawModel):
    def __init__(self, config: DiscreteRrlConfig) -> None:
        super().__init__()
        self.config = config
        exp_root = self.config.get_best_exp_root()
        if exp_root is None:
            raise FileNotFoundError(
                f'No experiment root found for dataset {self.config.dataset.name} '
                f'and group {self.config.train.group_names}!'
            )
        self.exp_root = exp_root

    def get_rrl(self) -> Rrl:
        return Rrl(self.config.get_rrl_file(self.exp_root))

    @override
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @override
    def predict(self, X: pd.DataFrame, y: pd.Series) -> ModelReturn:
        predicted = self.get_rrl().eval(X, prob=True)
        return predicted, {}

    def eval(self, data: pd.DataFrame) -> tuple[NDArray, Rrl]:
        rrl = self.get_rrl()
        result = rrl.eval(data)
        return result, rrl

    def interpret(self, data: pd.DataFrame) -> tuple[list[float], list[Line], Rrl]:
        rrl = self.get_rrl()
        result, active_lines = rrl.interpret(data)
        return result, active_lines, rrl
