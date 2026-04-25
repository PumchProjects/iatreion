from dataclasses import dataclass
from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from iatreion.configs import DataName, DiscreteRrlConfig, RrlEvalConfig
from iatreion.train_utils import make_data_labels
from iatreion.trainers import Recorder, TrainerReturn

from .rrl_eval_common import calc_score, deduplicate_by_keep, get_max_label
from .rrl_eval_data import build_model, get_data_model
from .rrl_eval_explain import get_sample_explanation


@dataclass(frozen=True)
class RrlTermOption:
    module: DataName
    kind: Literal['bias', 'rule']
    index: int | None
    label: str
    score: float
    rule: str

    @property
    def display_index(self) -> str:
        return 'Bias' if self.kind == 'bias' else f'#{self.index}'


def get_models(config: RrlEvalConfig) -> list[tuple[str, list[list[str]]]]:
    model = build_model(config)
    names = model.config.dataset.names
    models = model.get_models()
    rule_list: list[tuple[str, list[list[str]]]] = []
    for name, rrl in zip(names, models, strict=True):
        bias_label = get_max_label(rrl.biases, rrl.labels)
        bias_score = calc_score(rrl.biases)
        rules: list[list[str]] = [[bias_label, f'{bias_score:.2f}']]
        for line in rrl.lines:
            label = get_max_label(line.weights, line.labels)
            score = calc_score(line.weights)
            rules.append([label, f'{score:.2f}', line.print_rule()])
        rule_list.append((name, rules))
    return rule_list


def get_rule_options(config: RrlEvalConfig) -> list[RrlTermOption]:
    model = build_model(config)
    names = model.config.dataset.names
    models = model.get_models()
    options: list[RrlTermOption] = []
    for name, rrl in zip(names, models, strict=True):
        options.append(
            RrlTermOption(
                module=name,
                kind='bias',
                index=None,
                label=get_max_label(rrl.biases, rrl.labels),
                score=calc_score(rrl.biases),
                rule='Initial Bias',
            )
        )
        for index, line in enumerate(rrl.lines):
            options.append(
                RrlTermOption(
                    module=name,
                    kind='rule',
                    index=index,
                    label=get_max_label(line.weights, line.labels),
                    score=calc_score(line.weights),
                    rule=line.print_rule(),
                )
            )
    return options


def format_enabled_terms(config: RrlEvalConfig, names: list[DataName]) -> str:
    lines = ['Enabled RRL terms:']
    for name in names:
        bias_is_default = name not in config.enabled_biases
        rules_are_default = name not in config.enabled_rules
        if bias_is_default and rules_are_default:
            lines.append(f'{name}: all terms')
            continue

        bias = config.enabled_biases.get(name, True)
        rule_indices = config.enabled_rules.get(name)
        if rule_indices is None:
            rules = 'all rules'
        elif rule_indices:
            rules = 'rules ' + ', '.join(map(str, rule_indices))
        else:
            rules = 'no rules'
        lines.append(f'{name}: bias {"on" if bias else "off"}, {rules}')
    lines.append(f'Zero-mean fallback: {config.zero_mean_fallback}')
    return '\n'.join(lines)


def get_result(config: RrlEvalConfig) -> tuple[list[list[str]], ...]:
    sample = get_sample_explanation(config)
    result_list = [
        [
            sample.final_label,
            f'{sample.final_probability:.2%}',
            f'{sample.final_confidence:.2%}',
        ]
    ]
    pred_list = [
        [
            module.name,
            module.predicted_label,
            f'{module.predicted_probability:.2%}',
            f'{module.confidence:.2%}',
            f'{module.weight:.4f}',
        ]
        for module in sample.modules
    ]
    bias_list = [
        [module.name, module.bias_label, f'{module.bias_score:.2f}']
        for module in sample.modules
    ]
    support_list: list[list[str]] = []
    oppose_list: list[list[str]] = []
    if sample.final_label:
        for module in sample.modules:
            for rule in module.rules:
                row = [module.name, rule.label, f'{rule.score:.2f}', rule.rule]
                if rule.label == sample.final_label:
                    support_list.append(row)
                else:
                    oppose_list.append(row)
    return result_list, pred_list, bias_list, support_list, oppose_list


def get_batched_result(config: RrlEvalConfig) -> pd.DataFrame:
    data, additional_data, _, model = get_data_model(config)
    result, confidence = model.eval(data)
    y_pred = get_max_label(result)
    y_pred.name = 'Label'
    y_score = calc_score(result)
    y_score.name = 'Probability'
    confidence.name = 'Confidence'
    df = pd.concat(additional_data + [y_pred, y_score, confidence], axis=1)
    return deduplicate_by_keep(df, config.keep)


def get_eval_result(
    config: RrlEvalConfig,
) -> tuple[str, Figure | None, DiscreteRrlConfig]:
    data, _, group_names, model = get_data_model(config)
    assert group_names is not None
    result, _ = model.eval(
        data,
        enabled_biases=config.enabled_biases,
        enabled_rules=config.enabled_rules,
        zero_mean_fallback=config.zero_mean_fallback,
    )
    result = pd.concat([result, group_names], axis=1)
    train_config = model.config.train
    # Only select data in the target groups
    X_df, y_df = make_data_labels(result, train_config, group_names.columns.to_list())
    # Drop predictions that are failed
    y_df = y_df[~X_df.isna().all(axis=1)]
    X_df = X_df.dropna(how='all')
    y_true = y_df.map(train_config.get_group_index_mapping()).to_numpy()
    y_score = X_df.to_numpy()
    recorder = Recorder(train_config)
    eval_result = recorder.record(TrainerReturn(0.0, y_true, y_score))
    fig = recorder.roc.fig if train_config.plot_roc else None
    summary = format_enabled_terms(config, model.config.dataset.names)
    return f'{summary}\n\n{eval_result}', fig, model.config
