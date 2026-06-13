from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.figure import Figure

from iatreion.configs import DataName, DiscreteRrlConfig, RrlEvalConfig
from iatreion.train_utils import make_data_labels
from iatreion.trainers import record_evaluation
from iatreion.utils import write_spreadsheet

from .rrl_eval_common import calc_score, get_max_label
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


def _calc_odds_ratio(
    active_positive: int,
    active_negative: int,
    inactive_positive: int,
    inactive_negative: int,
) -> tuple[float, float, float, float, bool]:
    counts = np.array(
        [active_positive, active_negative, inactive_positive, inactive_negative],
        dtype=float,
    )
    if counts.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, False

    corrected = bool((counts == 0).any())
    calc_counts = counts + 0.5 if corrected else counts
    a, b, c, d = calc_counts
    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se = np.sqrt(np.sum(1 / calc_counts))
    ci_lower = np.exp(log_or - 1.96 * se)
    ci_upper = np.exp(log_or + 1.96 * se)
    _, pvalue = stats.fisher_exact(
        [[active_positive, active_negative], [inactive_positive, inactive_negative]]
    )
    return float(odds_ratio), float(ci_lower), float(ci_upper), float(pvalue), corrected


def _get_labeled_eval_target(
    data: list[pd.DataFrame],
    group_names: pd.DataFrame,
    model,
) -> pd.Series:
    result = model.eval(data)
    result = pd.concat([result, group_names], axis=1)
    train_config = model.config.train
    X_df, y_df = make_data_labels(result, train_config, group_names.columns.to_list())
    available = ~X_df.isna().all(axis=1)
    return y_df.loc[available]


def _prepare_module_data(
    frame: pd.DataFrame, index: pd.Index, rrl, *, keep: str
) -> tuple[pd.DataFrame, pd.Series]:
    frame = frame[~frame.index.duplicated(keep=keep)].reindex(index)
    available = ~frame.isna().all(axis=1)
    return rrl.impute(frame), available


def get_rule_or_table(config: RrlEvalConfig) -> pd.DataFrame:
    data, _, group_names, model = get_data_model(config)
    assert group_names is not None

    y_true = _get_labeled_eval_target(data, group_names, model)
    names = model.config.dataset.names
    models = model.get_models()
    rows: list[dict[str, object]] = []
    for name, frame, rrl in zip(names, data, models, strict=True):
        module_data, available = _prepare_module_data(
            frame, y_true.index, rrl, keep=config.keep
        )
        for rule_index, line in enumerate(rrl.lines):
            outcome_label = get_max_label(line.weights, line.labels)
            rule_eval = line.activation(module_data)
            valid = rule_eval.valid & available
            active = rule_eval.truth & valid
            outcome = y_true == outcome_label

            active_positive = int((active & outcome).sum())
            active_negative = int((active & ~outcome).sum())
            inactive_positive = int((valid & ~active & outcome).sum())
            inactive_negative = int((valid & ~active & ~outcome).sum())
            odds_ratio, ci_lower, ci_upper, pvalue, corrected = _calc_odds_ratio(
                active_positive,
                active_negative,
                inactive_positive,
                inactive_negative,
            )
            rows.append(
                {
                    'Module': name,
                    'Rule Index': rule_index,
                    'Rule Label': outcome_label,
                    'Outcome Label': outcome_label,
                    'Rule Score': calc_score(line.weights),
                    'Training Support': line.support,
                    'Mean Coverage': line.mean_coverage,
                    'Rule': line.print_rule(),
                    'N': int(y_true.shape[0]),
                    'N Valid': int(valid.sum()),
                    'N Active': int(active.sum()),
                    'Active Outcome+': active_positive,
                    'Active Outcome-': active_negative,
                    'Inactive Outcome+': inactive_positive,
                    'Inactive Outcome-': inactive_negative,
                    'OR': odds_ratio,
                    'OR 95% CI Lower': ci_lower,
                    'OR 95% CI Upper': ci_upper,
                    'Fisher p-value': pvalue,
                    'Haldane Correction': corrected,
                }
            )
    return pd.DataFrame(rows)


def save_batched_result_table(table: pd.DataFrame, path: str | Path) -> Path:
    return write_spreadsheet(path, table, float_format='%.4f')


def save_rule_or_table(table: pd.DataFrame, path: str | Path) -> Path:
    return write_spreadsheet(path, table, index=False, float_format='%.6g')


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


def get_result(
    config: RrlEvalConfig,
) -> tuple[
    str,
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[str]],
]:
    sample = get_sample_explanation(config)
    result_list = [
        [
            sample.final_label,
            f'{sample.final_score:+.2f}',
            f'{sample.final_boundary:+.2f}',
            f'{sample.final_probability:.2%}',
            f'{sample.positive_probability:.2%}',
            f'{sample.threshold:.2%}',
        ]
    ]
    pred_list = [
        [
            module.name,
            module.label,
            f'{module.score:+.2f}',
            f'{module.probability:.2%}',
            f'{module.weight:.4f}',
        ]
        for module in sample.modules
    ]
    pred_list.append(
        [
            'Total',
            sample.final_label,
            f'{sample.final_score:+.2f}',
            f'{sample.final_probability:.2%}',
            '1.0000',
        ]
    )
    bias_list = [
        [module.name, module.bias_label, f'{module.bias_signed_score:+.2f}']
        for module in sample.modules
    ]
    support_list: list[list[str]] = []
    oppose_list: list[list[str]] = []
    if sample.final_label:
        for module in sample.modules:
            for rule in module.rules:
                row = [module.name, rule.label, f'{rule.signed_score:+.2f}', rule.rule]
                if rule.signed_score >= 0:
                    support_list.append(row)
                else:
                    oppose_list.append(row)
    return (
        sample.sample_id,
        result_list,
        pred_list,
        bias_list,
        support_list,
        oppose_list,
    )


def get_batched_result(config: RrlEvalConfig) -> pd.DataFrame:
    data, additional_data, _, model = get_data_model(config)
    result = model.eval(data)
    y_pred = model.predict_labels(result)
    y_score = calc_score(result)
    y_score.name = 'Probability'
    df = pd.concat(additional_data + [y_pred, y_score], axis=1)
    return df


def get_eval_result(
    config: RrlEvalConfig,
) -> tuple[str, Figure | None, DiscreteRrlConfig]:
    data, _, group_names, model = get_data_model(config)
    assert group_names is not None
    result = model.eval(
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
    finish = record_evaluation(
        train_config,
        y_true,
        y_score,
        threshold=model.artifact.default_threshold,
    )
    fig = finish.roc if train_config.plot_roc else None
    summary = format_enabled_terms(config, model.config.dataset.names)
    return f'{summary}\n\n{finish.ci_result}', fig, model.config
