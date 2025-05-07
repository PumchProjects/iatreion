from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from iatreion.configs import TrainConfig
from iatreion.utils import logger


@dataclass
class Record[T]:
    auc: T
    acc: T
    precision: T
    recall: T
    f1: T
    complexity: T
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
            name=f'ROC fold {fold}',
            alpha=0.3,
            lw=1,
            ax=self.ax,
            plot_chance_level=(fold == self.config.n_splits),
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


class Recorder:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.result = Record[list[float]]([], [], [], [], [], [], None)
        self.roc = RecordROC(config)

    def record(self, y_true: ArrayLike, y_score: ArrayLike, complexity: float) -> None:
        y_pred = y_score
        if self.config.record_auc:
            y_score = np.asarray(y_score)
            y_pred = y_score.argmax(axis=1)
            y_pos_score = y_score[:, 1]
            self.result.auc.append(
                self.roc.record(y_true, y_pos_score, len(self.result.auc) + 1)
                if self.config.plot_roc
                else roc_auc_score(
                    y_true,
                    y_pos_score if self.config.num_class <= 2 else y_score,
                    average='macro',
                    multi_class='ovr',
                )
            )
        self.result.acc.append(accuracy_score(y_true, y_pred))
        self.result.precision.append(
            precision_score(y_true, y_pred, average='macro', zero_division=0)
        )
        self.result.recall.append(
            recall_score(y_true, y_pred, average='macro', zero_division=0)
        )
        self.result.f1.append(
            f1_score(y_true, y_pred, average='macro', zero_division=0)
        )
        self.result.complexity.append(complexity)
        if self.result.cm is None:
            self.result.cm = confusion_matrix(y_true, y_pred)
        else:
            self.result.cm += confusion_matrix(y_true, y_pred)
        if self.config.record_auc:
            logger.info(f'AUC: {self.result.auc[-1]:.2%}')
        logger.info(f'ACC: {self.result.acc[-1]:.2%}')
        logger.info(f'P:   {self.result.precision[-1]:.2%}')
        logger.info(f'R:   {self.result.recall[-1]:.2%}')
        logger.info(f'F1:  {self.result.f1[-1]:.2%}')
        logger.info(f'COM: {self.result.complexity[-1]:.4f}')

    def finish(self) -> Record[float]:
        final = Record(
            np.mean(self.result.auc).item() if self.config.record_auc else 0.0,
            np.mean(self.result.acc).item(),
            np.mean(self.result.precision).item(),
            np.mean(self.result.recall).item(),
            np.mean(self.result.f1).item(),
            np.mean(self.result.complexity).item(),
            self.result.cm,
        )
        logger.info(f'Confusion matrix:\n{final.cm}')
        if self.config.record_auc:
            if self.config.plot_roc:
                logger.info(f'INT AUC: {self.roc.finish():.2%}')
            logger.info(f'AVG AUC: {final.auc:.2%}')
        logger.info(f'AVG ACC: {final.acc:.2%}')
        logger.info(f'AVG P:   {final.precision:.2%}')
        logger.info(f'AVG R:   {final.recall:.2%}')
        logger.info(f'AVG F1:  {final.f1:.2%}')
        logger.info(f'AVG COM: {final.complexity:.4f}')
        return final
