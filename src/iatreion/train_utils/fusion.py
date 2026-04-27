from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from iatreion.configs import TrainConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import load_dict, save_dict

FUSION_ARTIFACT_FILE = 'available_fusion.toml'
_PROB_EPS = 1e-6


def clip_probability(y_pos_score: NDArray) -> NDArray:
    return np.clip(y_pos_score, _PROB_EPS, 1 - _PROB_EPS)


def logit(y_pos_score: NDArray) -> NDArray:
    clipped = clip_probability(y_pos_score)
    return np.log(clipped / (1 - clipped))


def sigmoid(logits: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-logits))


def get_clinical_recall_threshold(
    y_true: NDArray,
    y_pos_score: NDArray,
    *,
    target_label: int,
    target_recall: float,
) -> float:
    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                y_pos_score,
                np.nextafter(y_pos_score, 1.0),
            ]
        )
    )

    target_mask = y_true == target_label
    feasible = []
    for threshold in thresholds:
        y_pred = (y_pos_score >= threshold).astype(int)
        recall = np.mean(y_pred[target_mask] == target_label)
        if recall >= target_recall:
            feasible.append(threshold)

    if not feasible:
        return 0.5
    if target_label == 1:
        return max(feasible).item()
    return min(feasible).item()


@dataclass(frozen=True)
class ModalityCalibrator:
    slope: float
    intercept: float

    def calibrated_logit(self, y_pos_score: NDArray) -> NDArray:
        return self.slope * logit(y_pos_score) + self.intercept

    def calibrated_pos_score(self, y_pos_score: NDArray) -> NDArray:
        return sigmoid(self.calibrated_logit(y_pos_score))


@dataclass(frozen=True)
class AvailableFusionArtifact:
    names: list[str]
    labels: list[str]
    positive_label: str
    weights: dict[str, float]
    calibrators: dict[str, ModalityCalibrator]
    clinical_threshold_label: str
    clinical_threshold_recall: float
    clinical_threshold: float

    @classmethod
    def fit(
        cls,
        config: TrainConfig,
        names: list[str],
        y_true: NDArray,
        y_pos_score_list: list[NDArray],
        y_mask_list: list[NDArray],
    ) -> 'AvailableFusionArtifact':
        labels = [
            label
            for label, _ in sorted(
                config.get_group_index_mapping().items(), key=lambda item: item[1]
            )
        ]
        if len(labels) != 2:
            raise IatreionException(
                'Available-modality fusion currently requires binary labels.'
            )

        calibrators: dict[str, ModalityCalibrator] = {}
        for name, y_pos_score, y_mask in zip(
            names, y_pos_score_list, y_mask_list, strict=True
        ):
            available = ~y_mask.astype(bool)
            X = logit(y_pos_score[available]).reshape(-1, 1)
            calibrator = LogisticRegression(
                penalty='l2', C=1.0, random_state=42, solver='lbfgs'
            ).fit(X, y_true[available])
            calibrators[name] = ModalityCalibrator(
                slope=calibrator.coef_[0, 0].item(),
                intercept=calibrator.intercept_[0].item(),
            )

        weights = dict.fromkeys(names, 1 / len(names))
        artifact = cls(
            names=names,
            labels=labels,
            positive_label=labels[1],
            weights=weights,
            calibrators=calibrators,
            clinical_threshold_label=config.clinical_threshold_label,
            clinical_threshold_recall=config.clinical_threshold_recall,
            clinical_threshold=0.5,
        )
        y_pos_score = artifact.predict_pos_score(names, y_pos_score_list, y_mask_list)
        threshold = get_clinical_recall_threshold(
            y_true,
            y_pos_score,
            target_label=config.clinical_threshold_index,
            target_recall=config.clinical_threshold_recall,
        )
        return cls(
            names=artifact.names,
            labels=artifact.labels,
            positive_label=artifact.positive_label,
            weights=artifact.weights,
            calibrators=artifact.calibrators,
            clinical_threshold_label=artifact.clinical_threshold_label,
            clinical_threshold_recall=artifact.clinical_threshold_recall,
            clinical_threshold=threshold,
        )

    @classmethod
    def load(cls, path: Path) -> 'AvailableFusionArtifact':
        if not path.is_file():
            raise IatreionException(
                'Available-fusion artifact not found: $path. '
                'Run internal discrete RRL scoring before final/external evaluation.',
                path=str(path),
            )
        data = load_dict(path)
        data['calibrators'] = {
            name: ModalityCalibrator(**params)
            for name, params in data['calibrators'].items()
        }
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict(asdict(self), path)

    def predict_pos_score(
        self,
        names: list[str],
        y_pos_score_list: list[NDArray],
        y_mask_list: list[NDArray],
    ) -> NDArray:
        missing = np.column_stack(y_mask_list).astype(bool)
        calibrated_logits = np.column_stack(
            [
                self.calibrators[name].calibrated_logit(
                    np.where(y_mask, 0.5, y_pos_score)
                )
                for name, y_pos_score, y_mask in zip(
                    names, y_pos_score_list, missing.T, strict=True
                )
            ]
        )
        weights = np.array([self.weights[name] for name in names], dtype=float)
        effective_weights = np.where(missing, 0.0, weights)
        denominator = effective_weights.sum(axis=1)
        numerator = (calibrated_logits * effective_weights).sum(axis=1)
        fused_logit = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        return sigmoid(fused_logit)

    def predict_scores(
        self,
        names: list[str],
        y_pos_score_list: list[NDArray],
        y_mask_list: list[NDArray],
    ) -> NDArray:
        y_pos_score = self.predict_pos_score(names, y_pos_score_list, y_mask_list)
        return np.column_stack([1 - y_pos_score, y_pos_score])

    def normalized_weights(self, names: list[str]) -> dict[str, float]:
        total = sum(self.weights[name] for name in names)
        if total == 0:
            return dict.fromkeys(names, 0.0)
        return {name: self.weights[name] / total for name in names}

    def predict_indices(self, y_pos_score: NDArray) -> NDArray:
        return (y_pos_score >= self.clinical_threshold).astype(int)

    def predict_labels(self, y_pos_score: NDArray) -> NDArray:
        labels = np.asarray(self.labels)
        return labels[self.predict_indices(y_pos_score)]
