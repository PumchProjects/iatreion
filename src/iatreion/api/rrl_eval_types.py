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
    labels: tuple[str, ...]
    weight: float
    predicted_label: str
    predicted_probability: float
    target_probability: float
    confidence: float
    bias_label: str
    bias_score: float
    bias_signed_score: float
    target_margin: float
    rules: tuple[RuleExplanation, ...]


@dataclass(frozen=True)
class SampleExplanation:
    sample_id: str
    final_label: str
    final_probability: float
    final_confidence: float
    modules: tuple[ModuleExplanation, ...]


@dataclass(frozen=True)
class RrlWaterfallBundle:
    sample: SampleExplanation
    module_table: pd.DataFrame
    contribution_table: pd.DataFrame
