from dataclasses import dataclass, field
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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
from iatreion.utils import logger

type TrainerReturn = tuple[
    float, NDArray, NDArray, dict[str, float | tuple[float, str]]
]


@dataclass
class Record[T]:
    time: T
    auc: T
    acc: T
    precision: T
    recall: T
    f1: T
    sensitivity: T
    specificity: T
    complexity: dict[str, tuple[T, str]] = field(default_factory=dict)
    cm: NDArray | None = None


@dataclass
class Finish:
    config: TrainConfig
    result: str
    final: Record[float]
    roc: Figure | None = None

    def log(self, name: str) -> None:
        with self.config.logging(f'train_{name}'):
            logger.info(self.result)
        if self.roc is not None:
            self.roc.savefig(self.config.get_roc_file(f'roc_{name}'), dpi=300)


class RecordROC:
    def __init__(self, config: TrainConfig, *, is_inner: bool = False) -> None:
        self.config = config
        self.n_folds = config.n_inner_folds if is_inner else config.n_outer_folds
        self.show_legends = self.n_folds <= 5
        self.tprs: list[NDArray] = []
        self.aucs: list[float] = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.fig, ax = plt.subplots(figsize=(6, 6))
        self.ax = cast(Axes, ax)

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
        self.fig.tight_layout()

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
        self.fig.tight_layout()

        return mean_auc, self.fig


class Recorder:
    def __init__(self, config: TrainConfig, *, is_inner: bool = False) -> None:
        self.config = config
        self.result = Record[list[float]](*([] for _ in range(8)))  # type: ignore
        self.roc = RecordROC(config, is_inner=is_inner)
        self.calc_sen_and_spc = config.num_class == 2
        self.y_true_all: list[NDArray] = []
        self.y_score_all: list[NDArray] = []

    def record(self, results: TrainerReturn) -> str:
        training_time, y_true, y_score, complexity = results
        self.y_true_all.append(y_true)
        self.y_score_all.append(y_score)
        self.result.time.append(training_time)
        y_pred = y_score.argmax(axis=1)
        labels = list(range(self.config.num_class))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if self.result.cm is None:
            self.result.cm = cm
        else:
            self.result.cm += cm
        y_pos_score = y_score[:, 1] if y_score.shape[1] >= 2 else y_score.squeeze()
        self.result.auc.append(
            self.roc.record(y_true, y_pos_score)
            if self.config.plot_roc
            else roc_auc_score(
                y_true,
                y_pos_score if self.config.num_class <= 2 else y_score,
                average='macro',
                multi_class='ovr',
                labels=labels,
            )
        )
        self.result.acc.append(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=np.nan
        )
        self.result.precision.append(precision)
        self.result.recall.append(recall)
        self.result.f1.append(f1)
        if self.calc_sen_and_spc:
            self.result.sensitivity.append(
                recall_score(
                    y_true, y_pred, labels=labels, pos_label=0, zero_division=np.nan
                )
            )
            self.result.specificity.append(
                recall_score(
                    y_true, y_pred, labels=labels, pos_label=1, zero_division=np.nan
                )
            )
        width = 4
        for key, value in complexity.items():
            if isinstance(value, tuple):
                value, fmt = value
            else:
                fmt = '.4f'
            self.result.complexity.setdefault(key, ([], fmt))[0].append(value)
            width = max(width, len(key))
        result_lines = [
            f'Confusion matrix:\n{cm}\n',
            f'{"AUC":{width}} {self.result.auc[-1]:.2%}\n',
            f'{"ACC":{width}} {self.result.acc[-1]:.2%}\n',
            f'{"P":{width}} {self.result.precision[-1]:.2%}\n',
            f'{"R":{width}} {self.result.recall[-1]:.2%}\n',
            f'{"F1":{width}} {self.result.f1[-1]:.2%}\n',
            f'{"SEN":{width}} {self.result.sensitivity[-1]:.2%}\n'
            if self.calc_sen_and_spc
            else '',
            f'{"SPC":{width}} {self.result.specificity[-1]:.2%}\n'
            if self.calc_sen_and_spc
            else '',
            *(
                f'{key:{width}} {values[-1]:{fmt}}\n'
                for key, (values, fmt) in self.result.complexity.items()
            ),
            f'{"Time":{width}} {training_time:.3f}s',
        ]
        return ''.join(result_lines)

    def finish(self) -> Finish:
        complexity: dict[str, tuple[float, str]] = {}
        width = 4
        for key, (values, fmt) in self.result.complexity.items():
            complexity[key] = (np.nanmean(values).item(), fmt)
            width = max(width, len(key))
        final = Record(
            np.mean(self.result.time).item(),
            np.nanmean(self.result.auc).item(),
            np.mean(self.result.acc).item(),
            np.nanmean(self.result.precision).item(),
            np.nanmean(self.result.recall).item(),
            np.nanmean(self.result.f1).item(),
            np.nanmean(self.result.sensitivity).item(),
            np.nanmean(self.result.specificity).item(),
            complexity,
            self.result.cm,
        )
        auc, roc = self.roc.finish() if self.config.plot_roc else (0.0, None)
        result_lines = [
            f'Confusion matrix:\n{final.cm}\n',
            f'INT {"AUC":{width}} {auc:.2%}\n' if self.config.plot_roc else '',
            f'AVG {"AUC":{width}} {final.auc:.2%}\n',
            f'AVG {"ACC":{width}} {final.acc:.2%}\n',
            f'AVG {"P":{width}} {final.precision:.2%}\n',
            f'AVG {"R":{width}} {final.recall:.2%}\n',
            f'AVG {"F1":{width}} {final.f1:.2%}\n',
            f'AVG {"SEN":{width}} {final.sensitivity:.2%}\n'
            if self.calc_sen_and_spc
            else '',
            f'AVG {"SPC":{width}} {final.specificity:.2%}\n'
            if self.calc_sen_and_spc
            else '',
            *(
                f'AVG {key:{width}} {value:{fmt}}\n'
                for key, (value, fmt) in complexity.items()
            ),
            f'AVG {"Time":{width}} {final.time:.3f}s',
        ]
        return Finish(self.config, ''.join(result_lines), final, roc)
