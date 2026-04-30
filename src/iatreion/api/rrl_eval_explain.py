import numpy as np
import pandas as pd

from iatreion.configs import RrlEvalConfig
from iatreion.exceptions import IatreionException
from iatreion.models import Line, Rrl
from iatreion.train_utils.fusion import logit

from .rrl_eval_common import (
    calc_score,
    get_max_label,
    probability_for_label,
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
    labels: list[str],
    positive_label: str,
    threshold: float,
    names: list[str],
    models: list[Rrl],
    scores: list[pd.DataFrame],
    predictions: list[pd.DataFrame],
    active_lines: list[tuple[str, Line]],
    final_score: pd.DataFrame,
    result: pd.DataFrame,
) -> SampleExplanation:
    _validate_binary_labels(labels, positive_label)
    final_row = final_score.iloc[0]
    result_row = result.iloc[0]
    final_probability = probability_for_label(result_row, final_label)
    positive_probability = probability_for_label(result_row, positive_label)
    final_margin = signed_margin(final_row, labels, final_label)
    final_boundary = signed_boundary(threshold, final_label, positive_label)
    active_line_map: dict[str, list[Line]] = {name: [] for name in names}
    for name, line in active_lines:
        active_line_map[name].append(line)

    modules: list[ModuleExplanation] = []
    for name, rrl, score, pred in zip(names, models, scores, predictions, strict=True):
        _validate_binary_labels(rrl.labels, positive_label)
        score_row = score.iloc[0]
        pred_row = pred.iloc[0]
        module_score = signed_margin(score_row, rrl.labels, final_label)
        module_probability = probability_for_label(pred_row, final_label)
        bias_label = get_max_label(rrl.biases, rrl.labels)
        bias_score = calc_score(rrl.biases)
        bias_signed_score = signed_margin(rrl.biases, rrl.labels, final_label)
        rules = tuple(
            RuleExplanation(
                label=get_max_label(line.weights, line.labels),
                score=calc_score(line.weights),
                signed_score=signed_margin(
                    line.weights,
                    line.labels,
                    final_label,
                ),
                rule=line.print_rule(),
            )
            for line in active_line_map[name]
        )
        modules.append(
            ModuleExplanation(
                name=name,
                weight=rrl.weight,
                label=final_label,
                score=module_score,
                probability=module_probability,
                bias_label=bias_label,
                bias_score=bias_score,
                bias_signed_score=bias_signed_score,
                rules=rules,
            )
        )

    return SampleExplanation(
        sample_id=sample_id,
        labels=(labels[0], labels[1]),
        positive_label=positive_label,
        final_label=final_label,
        final_score=final_margin,
        final_boundary=final_boundary,
        final_probability=final_probability,
        positive_probability=positive_probability,
        threshold=threshold,
        modules=tuple(modules),
    )


def get_sample_explanation(config: RrlEvalConfig) -> SampleExplanation:
    data, _, _, model = get_data_model(config)
    full_result = model.eval(data)
    sample_id = resolve_sample_id(config, full_result)
    sample_data = select_sample_data(data, sample_id, keep=config.keep)
    (
        names,
        models,
        scores,
        predictions,
        active_lines,
        final_score,
        result,
    ) = model.interpret(sample_data)
    final_label = model.predict_labels(result).item()
    return build_sample_explanation(
        sample_id,
        final_label,
        model.artifact.labels,
        model.artifact.positive_label,
        model.artifact.clinical_threshold,
        names,
        models,
        scores,
        predictions,
        active_lines,
        final_score,
        result,
    )


def _validate_binary_labels(labels: list[str], positive_label: str) -> None:
    if len(labels) != 2 or positive_label not in labels:
        raise IatreionException(
            'RRL explanation currently requires binary labels containing '
            'positive label "$positive_label"; got [$labels].',
            positive_label=positive_label,
            labels=', '.join(labels),
        )


def other_label(labels: list[str] | tuple[str, str], label: str) -> str:
    for candidate in labels:
        if candidate != label:
            return candidate
    return f'not {label}'


def signed_margin(
    values: list[float] | pd.Series,
    labels: list[str] | tuple[str, str],
    target_label: str,
) -> float:
    other = other_label(labels, target_label)
    if isinstance(values, list):
        value_by_label = dict(zip(labels, values, strict=True))
        return value_by_label[target_label] - value_by_label[other]
    return float(values[target_label] - values[other])


def positive_boundary(threshold: float) -> float:
    threshold_array = np.array([threshold], dtype=float)
    return float(logit(threshold_array).item())


def signed_boundary(
    threshold: float,
    target_label: str,
    positive_label: str,
) -> float:
    boundary = positive_boundary(threshold)
    return boundary if target_label == positive_label else -boundary


def get_rule_waterfall_data(config: RrlEvalConfig) -> RrlWaterfallBundle:
    sample = get_sample_explanation(config)
    if not sample.final_label:
        raise IatreionException(
            'Cannot plot RRL waterfall for sample "$sample_id" because '
            'the final label is empty.',
            sample_id=sample.sample_id,
        )

    top_k = max(0, config.top_k)
    module_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    sample_id_text = sample.sample_id
    for module in sample.modules:
        for rule in module.rules:
            rule_rows.append(
                {
                    'Sample ID': sample_id_text,
                    'Final Label': sample.final_label,
                    'Final Probability': sample.final_probability,
                    'Positive Label': sample.positive_label,
                    'Positive Probability': sample.positive_probability,
                    'Threshold': sample.threshold,
                    'Boundary': sample.final_boundary,
                    'Module': module.name,
                    'Kind': 'Rule',
                    'Display': rule.rule,
                    'Label': rule.label,
                    'Score': rule.score,
                    'Signed Score': rule.signed_score,
                    'Abs Score': abs(rule.signed_score),
                    'Direction': 'Support' if rule.signed_score >= 0 else 'Oppose',
                }
            )

        module_rows.append(
            {
                'Sample ID': sample_id_text,
                'Final Label': sample.final_label,
                'Final Score': sample.final_score,
                'Boundary': sample.final_boundary,
                'Final Probability': sample.final_probability,
                'Positive Label': sample.positive_label,
                'Positive Probability': sample.positive_probability,
                'Threshold': sample.threshold,
                'Module': module.name,
                'Module Weight': module.weight,
                'Module Label': module.label,
                'Module Score': module.score,
                'Module Probability': module.probability,
                'Bias Label': module.bias_label,
                'Bias Score': module.bias_score,
                'Bias Signed Score': module.bias_signed_score,
                'Active Rule Count': len(module.rules),
            }
        )

    displayed_rules = sorted(rule_rows, key=lambda row: row['Abs Score'], reverse=True)
    hidden_rules = displayed_rules[top_k:]
    displayed_rules = displayed_rules[:top_k]
    if hidden_rules:
        hidden_total = sum(float(row['Signed Score']) for row in hidden_rules)
        displayed_rules.append(
            {
                'Sample ID': sample_id_text,
                'Final Label': sample.final_label,
                'Final Probability': sample.final_probability,
                'Positive Label': sample.positive_label,
                'Positive Probability': sample.positive_probability,
                'Threshold': sample.threshold,
                'Boundary': sample.final_boundary,
                'Module': 'All',
                'Kind': 'Other',
                'Display': f'{len(hidden_rules)} other active rules',
                'Label': (
                    sample.final_label
                    if hidden_total >= 0
                    else other_label(sample.labels, sample.final_label)
                ),
                'Score': abs(hidden_total),
                'Signed Score': hidden_total,
                'Abs Score': abs(hidden_total),
                'Direction': 'Support' if hidden_total >= 0 else 'Oppose',
            }
        )

    total_bias = sum(module.bias_signed_score for module in sample.modules)
    contribution_rows: list[dict[str, object]] = [
        {
            'Sample ID': sample_id_text,
            'Final Label': sample.final_label,
            'Final Score': sample.final_score,
            'Boundary': sample.final_boundary,
            'Final Probability': sample.final_probability,
            'Positive Label': sample.positive_label,
            'Positive Probability': sample.positive_probability,
            'Threshold': sample.threshold,
            'Module': 'All',
            'Kind': 'Bias',
            'Display': 'Initial biases',
            'Label': (
                sample.final_label
                if total_bias >= 0
                else other_label(sample.labels, sample.final_label)
            ),
            'Score': abs(total_bias),
            'Signed Score': total_bias,
            'Abs Score': abs(total_bias),
            'Direction': 'Support' if total_bias >= 0 else 'Oppose',
            'Order': 0,
            'Start': total_bias,
            'End': total_bias,
        }
    ]
    cumulative = total_bias
    for row in reversed(displayed_rules):
        row['Start'] = cumulative
        cumulative += float(row['Signed Score'])
        row['End'] = cumulative
    for order, row in enumerate(displayed_rules, start=1):
        row['Order'] = order
        contribution_rows.append(row)

    module_rows.append(
        {
            'Sample ID': sample_id_text,
            'Final Label': sample.final_label,
            'Final Score': sample.final_score,
            'Boundary': sample.final_boundary,
            'Final Probability': sample.final_probability,
            'Positive Label': sample.positive_label,
            'Positive Probability': sample.positive_probability,
            'Threshold': sample.threshold,
            'Module': 'Total',
            'Module Weight': 1.0,
            'Module Label': sample.final_label,
            'Module Score': sample.final_score,
            'Module Probability': sample.final_probability,
            'Bias Label': contribution_rows[0]['Label'],
            'Bias Score': contribution_rows[0]['Score'],
            'Bias Signed Score': total_bias,
            'Active Rule Count': sum(len(module.rules) for module in sample.modules),
        }
    )

    module_table = pd.DataFrame(module_rows)
    contribution_table = pd.DataFrame(contribution_rows)
    return RrlWaterfallBundle(sample, module_table, contribution_table)
