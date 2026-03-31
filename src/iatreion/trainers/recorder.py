from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from iatreion.configs import TrainConfig
from iatreion.utils import logger, task

type TrainerReturn = tuple[
    float, NDArray, NDArray, dict[str, float | tuple[float, str]]
]


@dataclass
class RunningRecord:
    time: list[float] = field(default_factory=list)
    y_true_all: list[NDArray] = field(default_factory=list)
    y_score_all: list[NDArray] = field(default_factory=list)
    metrics: defaultdict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    complexity: dict[str, tuple[list[float], str]] = field(default_factory=dict)
    cm: NDArray | None = None
    weights: list[list[float]] | None = None
    bias: list[float] | None = None


@dataclass
class FinalRecord:
    time: float
    y_true: NDArray
    y_score: NDArray
    metrics: dict[str, float]
    complexity: dict[str, tuple[float, str]]
    cm: NDArray | None = None
    weights: list[float] | None = None
    bias: float | None = None

    @property
    def auc(self) -> float:
        return self.metrics['AUC']

    @property
    def acc(self) -> float:
        return self.metrics['ACC']

    @property
    def precision(self) -> float:
        return self.metrics['P']

    @property
    def recall(self) -> float:
        return self.metrics['R']

    @property
    def f1(self) -> float:
        return self.metrics['F1']

    @property
    def sensitivity(self) -> float:
        return self.metrics.get('SEN', np.nan)

    @property
    def specificity(self) -> float:
        return self.metrics.get('SPC', np.nan)


@dataclass
class Finish:
    config: TrainConfig
    result: str
    ci_result: str
    running: RunningRecord
    final: FinalRecord
    ci: dict[str, tuple[float, float, float]] | None = None
    roc: Figure | None = None

    def log(self, name: str) -> None:
        display_name = name.removeprefix('all_').replace('_', ' ').title()
        if display_name[-1].isdigit():
            prefix, fold = display_name.rsplit(maxsplit=1)
            display_name = f'{prefix} (Fold {fold})'
        logger.info(f'[bold green]Finished {display_name}:', extra={'markup': True})
        with self.config.logging(self.config.get_avg_log_file(name)):
            logger.info(self.result)
        with self.config.logging(self.config.get_ci_log_file(name)):
            logger.info(self.ci_result)
        np.savez_compressed(
            self.config.get_results_file(name),
            y_true=self.final.y_true,
            y_score=self.final.y_score,
            **{metric: np.array(vals) for metric, vals in self.running.metrics.items()},
        )
        if self.roc is not None:
            self.roc.savefig(self.config.get_roc_file(name), dpi=300)
        if self.final.weights is not None and self.final.bias is not None:
            with self.config.logging(f'weights_{name}'):
                for weight in self.final.weights:
                    logger.debug(f'{weight:.4f}')
                logger.debug(f'{self.final.bias:.4f}')


class RecordROC:
    def __init__(self, config: TrainConfig, *, is_inner: bool = False) -> None:
        self.config = config
        self.n_folds = config.n_inner_folds if is_inner else config.n_outer_folds
        self.show_legends = self.n_folds <= 5
        self.tprs: list[NDArray] = []
        self.aucs: list[float] = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.fig, self.ax = plt.subplots(figsize=(6, 6), layout='constrained')

    def record_fold(self, y_true: NDArray, y_pos_score: NDArray) -> float:
        fold = len(self.aucs) + 1
        viz = RocCurveDisplay.from_predictions(
            y_true,
            y_pos_score,
            name=f'{"" if self.show_legends else "_"}ROC fold {fold}',
            alpha=0.1,
            lw=1,
            ax=self.ax,
            plot_chance_level=(fold == self.n_folds),
        )
        interp_tpr = np.interp(self.mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(viz.roc_auc)
        return viz.roc_auc

    def record_final(self, y_true: NDArray, y_pos_score: NDArray) -> float:
        viz = RocCurveDisplay.from_predictions(
            y_true,
            y_pos_score,
            name='_ROC',
            color='b',
            alpha=0.8,
            lw=2,
            ax=self.ax,
            plot_chance_level=True,
        )

        self.ax.set(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title='ROC curve',
        )
        self.ax.legend(loc='lower right')

        self.aucs.append(viz.roc_auc)
        return viz.roc_auc

    def record(self, y_true: NDArray, y_pos_score: NDArray) -> float:
        if self.config.final:
            return self.record_final(y_true, y_pos_score)
        return self.record_fold(y_true, y_pos_score)

    def finish(self) -> tuple[float, Figure]:
        mean_tpr = np.nanmean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.nanstd(self.aucs)
        self.ax.plot(
            self.mean_fpr,
            mean_tpr,
            color='b',
            label=rf'Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})',
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.nanstd(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(
            self.mean_fpr,
            tprs_lower,
            tprs_upper,
            color='grey',
            alpha=0.2,
            label=r'$\pm$ 1 std. dev.',
        )

        self.ax.set(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title='Mean ROC curve with cross validation',
        )
        self.ax.legend(loc='lower right')

        return mean_auc, self.fig


class RecordFormatter:
    @staticmethod
    def _format_percentage(value: float) -> str:
        return 'nan' if np.isnan(value) else f'{value:.2%}'

    def _format_metrics(self, metrics: dict[str, float], width: int) -> list[str]:
        return [
            f'{metric:{width}} {self._format_percentage(value)}\n'
            for metric, value in metrics.items()
        ]

    @staticmethod
    def _format_complexity(
        complexity: dict[str, tuple[float, str]], width: int
    ) -> list[str]:
        return [
            f'{key:{width}} {value:{fmt}}\n' for key, (value, fmt) in complexity.items()
        ]

    def _format_mean_std(
        self, mean_std: dict[str, tuple[float, float]], width: int
    ) -> list[str]:
        return [
            f'{metric:{width}} {self._format_percentage(mean)} '
            f'± {self._format_percentage(std)}\n'
            for metric, (mean, std) in mean_std.items()
        ]

    def _format_bootstrap_ci(
        self, ci: dict[str, tuple[float, float, float]], width: int
    ) -> list[str]:
        return [
            f'{metric:{width}} {self._format_percentage(point)} '
            f'[{self._format_percentage(lower)}, {self._format_percentage(upper)}]\n'
            for metric, (point, lower, upper) in ci.items()
        ]

    def _format_time(self, time: float, width: int) -> str:
        return f'{"Time":{width}} {time:.3f}s'

    def _format(
        self,
        cm: NDArray,
        complexity: dict[str, tuple[float, str]],
        training_time: float,
        width: int,
        *metrics_lines: str,
    ) -> str:
        result_lines = [
            f'Confusion matrix:\n{cm}\n',
            *metrics_lines,
            *self._format_complexity(complexity, width),
            self._format_time(training_time, width),
        ]
        return ''.join(result_lines)

    def format_fold(
        self,
        *,
        cm: NDArray,
        metrics: dict[str, float],
        complexity: dict[str, tuple[float, str]],
        training_time: float,
        width: int,
    ) -> str:
        return self._format(
            cm, complexity, training_time, width, *self._format_metrics(metrics, width)
        )

    def format_final_avg(
        self,
        *,
        final: FinalRecord,
        mean_std: dict[str, tuple[float, float]],
        auc: float | None,
        width: int,
    ) -> str:
        return self._format(
            final.cm,
            final.complexity,
            final.time,
            width,
            '' if auc is None else f'Interpolated AUC {self._format_percentage(auc)}\n',
            *self._format_mean_std(mean_std, width),
        )

    def format_final_ci(
        self,
        *,
        final: FinalRecord,
        ci: dict[str, tuple[float, float, float]],
        width: int,
    ) -> str:
        return self._format(
            final.cm,
            final.complexity,
            final.time,
            width,
            *self._format_bootstrap_ci(ci, width),
        )

    def format_final_metrics(self, *, final: FinalRecord, width: int) -> str:
        return self._format(
            final.cm,
            final.complexity,
            final.time,
            width,
            *self._format_metrics(final.metrics, width),
        )


class Recorder:
    def __init__(self, config: TrainConfig, *, is_inner: bool = False) -> None:
        self.config = config
        self.roc = RecordROC(config, is_inner=is_inner)
        self.calc_sen_and_spc = config.num_class == 2
        self.result = RunningRecord()
        self.formatter = RecordFormatter()

    def _get_labels(self) -> list[int]:
        return list(range(self.config.num_class))

    @staticmethod
    def _predict_from_score(y_score: NDArray) -> tuple[NDArray, NDArray]:
        y_pred = y_score.argmax(axis=1)
        y_pos_score = y_score[:, 1] if y_score.shape[1] >= 2 else y_score.squeeze()
        return y_pred, y_pos_score

    def _calc_auc(
        self, y_true: NDArray, y_score: NDArray, y_pos_score: NDArray, labels: list[int]
    ) -> float:
        try:
            return roc_auc_score(
                y_true,
                y_pos_score if self.config.num_class <= 2 else y_score,
                average='macro',
                multi_class='ovr',
                labels=labels,
            )
        except ValueError:
            return np.nan

    def _calc_metrics(
        self, y_true: NDArray, y_pred: NDArray, auc: float, labels: list[int]
    ) -> dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=np.nan
        )
        metrics = {
            'AUC': auc,
            'ACC': accuracy_score(y_true, y_pred),
            'P': precision,
            'R': recall,
            'F1': f1,
        }
        if self.calc_sen_and_spc:
            metrics['SEN'] = recall_score(
                y_true, y_pred, labels=labels, pos_label=0, zero_division=np.nan
            )
            metrics['SPC'] = recall_score(
                y_true, y_pred, labels=labels, pos_label=1, zero_division=np.nan
            )
        return metrics

    def _calc_metrics_from_prediction(
        self, y_true: NDArray, y_score: NDArray
    ) -> dict[str, float]:
        labels = self._get_labels()
        y_pred, y_pos_score = self._predict_from_score(y_score)
        auc = self._calc_auc(y_true, y_score, y_pos_score, labels)
        return self._calc_metrics(y_true, y_pred, auc, labels)

    def _append_fold_metrics(self, metrics: dict[str, float]) -> None:
        for metric, value in metrics.items():
            self.result.metrics[metric].append(value)

    def _record_complexity(
        self, complexity: dict[str, float | tuple[float, str]]
    ) -> tuple[dict[str, tuple[float, str]], int]:
        width = 4
        com = {
            key: value if isinstance(value, tuple) else (value, '.4f')
            for key, value in complexity.items()
        }
        for key, (value, fmt) in com.items():
            self.result.complexity.setdefault(key, ([], fmt))[0].append(value)
            width = max(width, len(key))
        return com, width

    def _update_confusion_matrix(self, cm: NDArray) -> None:
        if self.result.cm is None:
            self.result.cm = cm
        else:
            self.result.cm += cm

    def _summarize_complexity(self) -> tuple[dict[str, tuple[float, str]], int]:
        complexity: dict[str, tuple[float, str]] = {}
        width = 4
        for key, (values, fmt) in self.result.complexity.items():
            complexity[key] = (np.nanmean(values).item(), fmt)
            width = max(width, len(key))
        return complexity, width

    def _summarize_fold_metrics(self) -> dict[str, tuple[float, float]]:
        mean_std: dict[str, tuple[float, float]] = {}
        for metric, values in self.result.metrics.items():
            arr = np.asarray(values, dtype=float)
            mean_std[metric] = (np.nanmean(arr).item(), np.nanstd(arr).item())
        return mean_std

    def _build_final_record(
        self,
        complexity: dict[str, tuple[float, str]],
        y_true: NDArray,
        y_score: NDArray,
        point_estimates: dict[str, float],
    ) -> FinalRecord:
        return FinalRecord(
            np.mean(self.result.time).item(),
            y_true,
            y_score,
            point_estimates,
            complexity,
            self.result.cm,
            None
            if self.result.weights is None
            else np.mean(self.result.weights, axis=0).tolist(),
            None if self.result.bias is None else np.mean(self.result.bias).item(),
        )

    def _calc_bootstrap_ci(
        self, y_true: NDArray, y_score: NDArray, point_estimates: dict[str, float]
    ) -> dict[str, tuple[float, float, float]]:
        sample_count = y_true.shape[0]
        if sample_count == 0:
            return {
                metric: (value, np.nan, np.nan)
                for metric, value in point_estimates.items()
            }

        rng = np.random.default_rng(self.config.seed)
        metric_samples = {metric: [] for metric in point_estimates}
        index = np.arange(sample_count)
        with task('Bootstrap:', self.config.bootstrap_samples) as bootstrap_advance:
            for _ in range(self.config.bootstrap_samples):
                sampled_index = rng.choice(index, size=sample_count, replace=True)
                sampled_metrics = self._calc_metrics_from_prediction(
                    y_true[sampled_index], y_score[sampled_index]
                )
                for metric, value in sampled_metrics.items():
                    metric_samples[metric].append(value)
                bootstrap_advance()

        alpha = 1 - self.config.ci_level
        q_lower = 100 * alpha / 2
        q_upper = 100 * (1 - alpha / 2)
        ci: dict[str, tuple[float, float, float]] = {}
        for metric, point in point_estimates.items():
            values = np.asarray(metric_samples[metric], dtype=float)
            if np.isnan(values).all():
                ci[metric] = (point, np.nan, np.nan)
                continue
            ci[metric] = (
                point,
                np.nanpercentile(values, q_lower).item(),
                np.nanpercentile(values, q_upper).item(),
            )
        return ci

    def record(self, results: TrainerReturn) -> str:
        training_time, y_true, y_score, complexity = results
        self.result.y_true_all.append(y_true)
        self.result.y_score_all.append(y_score)
        self.result.time.append(training_time)

        labels = self._get_labels()
        y_pred, y_pos_score = self._predict_from_score(y_score)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        self._update_confusion_matrix(cm)
        fold_auc = (
            self.roc.record(y_true, y_pos_score)
            if self.config.plot_roc
            else self._calc_auc(y_true, y_score, y_pos_score, labels)
        )
        metrics = self._calc_metrics(y_true, y_pred, fold_auc, labels)
        self._append_fold_metrics(metrics)
        com, width = self._record_complexity(complexity)
        return self.formatter.format_fold(
            cm=cm,
            metrics=metrics,
            complexity=com,
            training_time=training_time,
            width=width,
        )

    def record_weights_and_bias(self, weights: list[float], bias: float) -> None:
        if self.result.weights is None:
            self.result.weights = []
        if self.result.bias is None:
            self.result.bias = []
        self.result.weights.append(weights)
        self.result.bias.append(bias)

    def finish(self, *, calc_ci: bool = True) -> Finish:
        complexity, width = self._summarize_complexity()
        mean_std = self._summarize_fold_metrics()
        y_true = np.concatenate(self.result.y_true_all)
        y_score = np.concatenate(self.result.y_score_all)
        point_estimates = self._calc_metrics_from_prediction(y_true, y_score)
        final = self._build_final_record(complexity, y_true, y_score, point_estimates)
        auc, roc = self.roc.finish() if self.config.plot_roc else (None, None)
        result = self.formatter.format_final_avg(
            final=final, mean_std=mean_std, auc=auc, width=width
        )
        ci = None
        if calc_ci:
            ci = self._calc_bootstrap_ci(y_true, y_score, point_estimates)
            ci_result = self.formatter.format_final_ci(final=final, ci=ci, width=width)
        else:
            ci_result = self.formatter.format_final_metrics(final=final, width=width)
        return Finish(self.config, result, ci_result, self.result, final, ci, roc)
