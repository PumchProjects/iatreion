import numpy as np
import pandas as pd

from iatreion.configs import RrlEvalConfig, RrlEvalPlotConfig
from iatreion.exceptions import IatreionException
from iatreion.models import Line, Rrl

from .rrl_eval_common import (
    calc_score,
    calc_signed_score,
    get_max_label,
    opposing_label,
    probability_for_label,
    series_item,
)
from .rrl_eval_data import get_data_model
from .rrl_eval_types import (
    ModuleExplanation,
    RrlWaterfallBundle,
    RuleExplanation,
    SampleExplanation,
)


def get_requested_sample_id(config: RrlEvalConfig, result: pd.DataFrame) -> str:
    requested = config.sample_id.strip()
    if not requested:
        requested = str(result.index[0])
    return requested


def resolve_sample_id(config: RrlEvalConfig, result: pd.DataFrame) -> str:
    requested = get_requested_sample_id(config, result)

    index_text = result.index.map(str)
    if (index_text == requested).any():
        return requested

    available = result.index.astype(str).tolist()
    preview = ', '.join(available[:5]) or '(none)'
    raise IatreionException(
        'Unknown RRL sample ID "$sample_id". First available IDs: $preview',
        sample_id=requested,
        preview=preview,
    )


def make_missing_sample(frame: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    return frame.iloc[:0].reindex(
        pd.Index([sample_id], name=frame.index.name),
    )


def select_sample_frame(
    frame: pd.DataFrame,
    sample_id: str,
    *,
    keep: str,
) -> pd.DataFrame:
    matches = frame.loc[frame.index.map(str) == sample_id]
    if matches.empty:
        return make_missing_sample(frame, sample_id)
    if keep == 'first':
        return matches.iloc[[0]]
    return matches.iloc[[-1]]


def select_sample_data(
    data: list[pd.DataFrame],
    sample_id: str,
    *,
    keep: str,
) -> list[pd.DataFrame]:
    return [select_sample_frame(frame, sample_id, keep=keep) for frame in data]


def build_sample_explanation(
    sample_id: str,
    final_label: str,
    names: list[str],
    models: list[Rrl],
    predictions: list[tuple[pd.DataFrame, pd.Series]],
    active_lines: list[tuple[str, Line]],
    result: pd.DataFrame,
    confidence: pd.Series,
) -> SampleExplanation:
    final_probability = series_item(calc_score(result))
    final_confidence = series_item(confidence)
    active_line_map: dict[str, list[Line]] = {name: [] for name in names}
    for name, line in active_lines:
        active_line_map[name].append(line)

    modules: list[ModuleExplanation] = []
    for name, rrl, (pred, conf) in zip(names, models, predictions, strict=True):
        pred_label = get_max_label(pred).item()
        pred_probability = series_item(calc_score(pred))
        pred_row = pred.iloc[0]
        target_probability = probability_for_label(pred_row, final_label)
        bias_label = get_max_label(rrl.biases, rrl.labels)
        bias_score = calc_score(rrl.biases)
        bias_signed_score = calc_signed_score(rrl.biases, rrl.labels, final_label)
        rules = tuple(
            RuleExplanation(
                label=get_max_label(line.weights, line.labels),
                score=calc_score(line.weights),
                signed_score=calc_signed_score(
                    line.weights,
                    line.labels,
                    final_label,
                ),
                rule=line.print_rule(),
            )
            for line in active_line_map[name]
        )
        target_margin = float('nan')
        if not np.isnan(bias_signed_score) and all(
            not np.isnan(rule.signed_score) for rule in rules
        ):
            target_margin = bias_signed_score + sum(rule.signed_score for rule in rules)
        modules.append(
            ModuleExplanation(
                name=name,
                labels=tuple(rrl.labels),
                weight=rrl.weight,
                predicted_label=pred_label,
                predicted_probability=pred_probability,
                target_probability=target_probability,
                confidence=series_item(conf),
                bias_label=bias_label,
                bias_score=bias_score,
                bias_signed_score=bias_signed_score,
                target_margin=target_margin,
                rules=rules,
            )
        )

    return SampleExplanation(
        sample_id=sample_id,
        final_label=final_label,
        final_probability=final_probability,
        final_confidence=final_confidence,
        modules=tuple(modules),
    )


def get_sample_explanation(config: RrlEvalConfig) -> SampleExplanation:
    data, _, _, model = get_data_model(config)
    full_result, _ = model.eval(data)
    sample_id = resolve_sample_id(config, full_result)
    sample_data = select_sample_data(data, sample_id, keep=config.keep)
    names, models, predictions, active_lines, result, confidence = model.interpret(
        sample_data
    )
    final_label = model.predict_labels(result).item()
    return build_sample_explanation(
        sample_id,
        final_label,
        names,
        models,
        predictions,
        active_lines,
        result,
        confidence,
    )


def get_rule_waterfall_data(config: RrlEvalPlotConfig) -> RrlWaterfallBundle:
    sample = get_sample_explanation(config)
    if not sample.final_label:
        raise IatreionException(
            'Cannot plot RRL waterfall for sample "$sample_id" because '
            'the final label is empty.',
            sample_id=sample.sample_id,
        )

    top_k = max(0, config.top_k)
    module_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    sample_id_text = sample.sample_id
    for module in sample.modules:
        if len(module.labels) != 2 or np.isnan(module.target_margin):
            raise IatreionException(
                'RRL waterfall currently requires binary classification; '
                'module "$name" has labels [$labels].',
                name=module.name,
                labels=', '.join(module.labels),
            )

        rules = sorted(
            module.rules,
            key=lambda rule: abs(rule.signed_score),
            reverse=True,
        )
        display_rules = list(rules[:top_k])
        hidden_rules = rules[top_k:]
        if hidden_rules:
            hidden_total = sum(rule.signed_score for rule in hidden_rules)
            display_rules.append(
                RuleExplanation(
                    label=(
                        sample.final_label
                        if hidden_total >= 0
                        else opposing_label(module.labels, sample.final_label)
                    ),
                    score=abs(hidden_total),
                    signed_score=hidden_total,
                    rule=f'{len(hidden_rules)} other active rules',
                )
            )

        rule_rows = [
            {
                'Kind': (
                    'Other' if rule.rule.endswith('other active rules') else 'Rule'
                ),
                'Display': rule.rule,
                'Label': rule.label,
                'Score': rule.score,
                'Signed Score': rule.signed_score,
            }
            for rule in display_rules
        ]
        cumulative = module.bias_signed_score
        for row in reversed(rule_rows):
            row['Start'] = cumulative
            row['End'] = cumulative + row['Signed Score']
            cumulative = row['End']

        contribution_rows.append(
            {
                'Sample ID': sample_id_text,
                'Final Label': sample.final_label,
                'Final Probability': sample.final_probability,
                'Final Confidence': sample.final_confidence,
                'Module': module.name,
                'Kind': 'Bias',
                'Display': 'Initial Bias',
                'Label': module.bias_label,
                'Score': module.bias_score,
                'Signed Score': module.bias_signed_score,
                'Abs Score': abs(module.bias_signed_score),
                'Direction': ('Support' if module.bias_signed_score >= 0 else 'Oppose'),
                'Order': 0,
                'Start': module.bias_signed_score,
                'End': module.bias_signed_score,
            }
        )
        for order, row in enumerate(rule_rows, start=1):
            contribution_rows.append(
                {
                    'Sample ID': sample_id_text,
                    'Final Label': sample.final_label,
                    'Final Probability': sample.final_probability,
                    'Final Confidence': sample.final_confidence,
                    'Module': module.name,
                    'Kind': row['Kind'],
                    'Display': row['Display'],
                    'Label': row['Label'],
                    'Score': row['Score'],
                    'Signed Score': row['Signed Score'],
                    'Abs Score': abs(row['Signed Score']),
                    'Direction': ('Support' if row['Signed Score'] >= 0 else 'Oppose'),
                    'Order': order,
                    'Start': row['Start'],
                    'End': row['End'],
                }
            )

        module_rows.append(
            {
                'Sample ID': sample_id_text,
                'Final Label': sample.final_label,
                'Final Probability': sample.final_probability,
                'Final Confidence': sample.final_confidence,
                'Module': module.name,
                'Module Weight': module.weight,
                'Module Label': module.predicted_label,
                'Module Probability': module.predicted_probability,
                'Target Probability': module.target_probability,
                'Confidence': module.confidence,
                'Bias Label': module.bias_label,
                'Bias Score': module.bias_score,
                'Bias Signed Score': module.bias_signed_score,
                'Target Margin': module.target_margin,
                'Active Rule Count': len(module.rules),
                'Displayed Rule Count': len(display_rules),
            }
        )

    module_table = pd.DataFrame(module_rows)
    contribution_table = pd.DataFrame(contribution_rows)
    return RrlWaterfallBundle(sample, module_table, contribution_table)
