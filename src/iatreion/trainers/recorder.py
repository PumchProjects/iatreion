from dataclasses import dataclass, field
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
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

type TrainerReturn = tuple[float, NDArray, NDArray, NDArray, dict[str, float]]


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
    consistency: T
    complexity: dict[str, T] = field(default_factory=dict)
    cm: NDArray | None = None


class RecordROC:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.tprs: list[NDArray] = []
        self.aucs: list[float] = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.fig, ax = plt.subplots(figsize=(6, 6))
        self.ax = cast(Axes, ax)

    def record(self, y_true: NDArray, y_pos_score: NDArray, fold: int) -> float:
        viz = RocCurveDisplay.from_predictions(
            y_true,
            y_pos_score,
            name=f'_ROC fold {fold}',
            alpha=0.1,
            lw=1,
            ax=self.ax,
            plot_chance_level=(fold == self.config.n_folds),
        )
        interp_tpr = np.interp(self.mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(viz.roc_auc)
        return viz.roc_auc

    def finish(self) -> float:
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        self.ax.plot(
            self.mean_fpr,
            mean_tpr,
            color='b',
            label=rf'Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})',
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.tprs, axis=0)
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
        self.fig.savefig(self.config.roc_file, dpi=300)

        return mean_auc


def consistency_ratio(y_pred: NDArray, index: NDArray) -> float:
    pred = pd.Series(y_pred, index=index)
    valid_pred = pred.groupby(level=0).filter(lambda x: len(x) > 1)
    if valid_pred.empty:
        return float('nan')
    nunique = valid_pred.groupby(level=0).nunique()
    return nunique[nunique == 1].sum() / len(nunique)


class Recorder:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.result = Record[list[float]](*([] for _ in range(9)))  # type: ignore
        self.roc = RecordROC(config)

    def record(self, results: TrainerReturn) -> None:
        training_time, y_true, y_score, index, complexity = results
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
            self.roc.record(y_true, y_pos_score, len(self.result.auc) + 1)
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
        self.result.consistency.append(consistency_ratio(y_pred, index))
        width = 4
        for key, value in complexity.items():
            self.result.complexity.setdefault(key, []).append(value)
            width = max(width, len(key))
        logger.info(f'Confusion matrix:\n{cm}')
        logger.info(f'{"AUC":{width}} {self.result.auc[-1]:.2%}')
        logger.info(f'{"ACC":{width}} {self.result.acc[-1]:.2%}')
        logger.info(f'{"P":{width}} {self.result.precision[-1]:.2%}')
        logger.info(f'{"R":{width}} {self.result.recall[-1]:.2%}')
        logger.info(f'{"F1":{width}} {self.result.f1[-1]:.2%}')
        if self.config.num_class == 2:
            logger.info(f'{"SEN":{width}} {self.result.sensitivity[-1]:.2%}')
            logger.info(f'{"SPC":{width}} {self.result.specificity[-1]:.2%}')
        logger.info(f'{"CST":{width}} {self.result.consistency[-1]:.2%}')
        for key, values in self.result.complexity.items():
            logger.info(f'{key:{width}} {values[-1]:.4f}')
        logger.info(f'{"Time":{width}} {training_time:.3f}s')

    def finish(self) -> Record[float]:
        complexity = {}
        width = 4
        for key, values in self.result.complexity.items():
            complexity[key] = np.nanmean(values).item()
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
            np.nanmean(self.result.consistency).item(),
            complexity,
            self.result.cm,
        )
        logger.info(f'Confusion matrix:\n{final.cm}')
        if self.config.plot_roc:
            logger.info(f'INT {"AUC":{width}} {self.roc.finish():.2%}')
        logger.info(f'AVG {"AUC":{width}} {final.auc:.2%}')
        logger.info(f'AVG {"ACC":{width}} {final.acc:.2%}')
        logger.info(f'AVG {"P":{width}} {final.precision:.2%}')
        logger.info(f'AVG {"R":{width}} {final.recall:.2%}')
        logger.info(f'AVG {"F1":{width}} {final.f1:.2%}')
        if self.config.num_class == 2:
            logger.info(f'AVG {"SEN":{width}} {final.sensitivity:.2%}')
            logger.info(f'AVG {"SPC":{width}} {final.specificity:.2%}')
        logger.info(f'AVG {"CST":{width}} {final.consistency:.2%}')
        for key, value in complexity.items():
            logger.info(f'AVG {key:{width}} {value:.4f}')
        logger.info(f'AVG {"Time":{width}} {final.time:.3f}s')
        return final
