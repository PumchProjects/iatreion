from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RuleExplanation:
    label: str
    score: float
    signed_score: float
    rule: str


@dataclass(frozen=True)
class ModuleExplanation:
    name: str
    weight: float
    label: str
    score: float
    probability: float
    bias_label: str
    bias_score: float
    bias_signed_score: float
    rules: tuple[RuleExplanation, ...]


@dataclass(frozen=True)
class SampleExplanation:
    sample_id: str
    labels: tuple[str, str]
    positive_label: str
    final_label: str
    final_score: float
    final_boundary: float
    final_probability: float
    positive_probability: float
    threshold: float
    modules: tuple[ModuleExplanation, ...]


@dataclass(frozen=True)
class RrlWaterfallBundle:
    sample: SampleExplanation
    module_table: pd.DataFrame
    contribution_table: pd.DataFrame
