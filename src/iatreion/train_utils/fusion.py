from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from iatreion.configs import TrainConfig
from iatreion.exceptions import IatreionException
from iatreion.utils import load_dict, save_dict

FUSION_ARTIFACT_FILE = 'available_fusion.toml'
_PROB_EPS = 1e-6
_FUSION_SCHEMA_VERSION = 2
_FUSION_POLICY = 'learned_global_weights'
_WEIGHT_OBJECTIVE = 'log_loss'


def clip_probability(y_pos_score: NDArray) -> NDArray:
    return np.clip(y_pos_score, _PROB_EPS, 1 - _PROB_EPS)


def logit(y_pos_score: NDArray) -> NDArray:
    clipped = clip_probability(y_pos_score)
    return np.log(clipped / (1 - clipped))


def sigmoid(logits: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-logits))


def binary_log_loss_from_logits(y_true: NDArray, logits: NDArray) -> float:
    y = y_true.astype(float)
    return np.mean(np.logaddexp(0.0, logits) - y * logits).item()


def get_threshold_candidates(y_pos_score: NDArray) -> NDArray:
    return np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0]),
                y_pos_score,
                np.nextafter(y_pos_score, 1.0),
            ]
        )
    )


def get_clinical_recall_threshold(
    y_true: NDArray,
    y_pos_score: NDArray,
    *,
    target_label: int,
    target_recall: float,
) -> float:
    thresholds = get_threshold_candidates(y_pos_score)
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


def get_youden_threshold(y_true: NDArray, y_pos_score: NDArray) -> float:
    thresholds = get_threshold_candidates(y_pos_score)
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    best_threshold = 0.5
    best_youden = -np.inf
    for threshold in thresholds:
        y_pred = y_pos_score >= threshold
        sensitivity = np.mean(y_pred[positive_mask])
        specificity = np.mean(~y_pred[negative_mask])
        youden = sensitivity + specificity - 1
        if youden > best_youden:
            best_threshold = threshold
            best_youden = youden
    return best_threshold.item()


def get_default_threshold_name(config: TrainConfig) -> str:
    return 'clinical_recall' if config.use_clinical_threshold else 'youden'


def get_operating_thresholds(
    config: TrainConfig, y_true: NDArray, y_pos_score: NDArray
) -> dict[str, float]:
    thresholds = {}
    if config.use_clinical_threshold:
        thresholds['clinical_recall'] = get_clinical_recall_threshold(
            y_true,
            y_pos_score,
            target_label=config.clinical_threshold_index,
            target_recall=config.clinical_threshold_recall,
        )
    thresholds['youden'] = get_youden_threshold(y_true, y_pos_score)
    return thresholds


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
    fusion_schema_version: int
    fusion_policy: str
    weight_objective: str
    names: list[str]
    labels: list[str]
    positive_label: str
    weights: dict[str, float]
    calibrators: dict[str, ModalityCalibrator]
    thresholds: dict[str, float]
    default_threshold_name: str

    @classmethod
    def fit(
        cls,
        config: TrainConfig,
        names: list[str],
        y_true: NDArray,
        y_pos_score_list: list[NDArray],
        y_mask_list: list[NDArray],
    ) -> 'AvailableFusionArtifact':
        labels = config.group_labels
        if len(labels) != 2:
            raise IatreionException(
                'Available-modality fusion currently requires binary labels.'
            )

        missing = np.column_stack(y_mask_list).astype(bool)
        calibrators = cls.fit_calibrators(names, y_true, y_pos_score_list, y_mask_list)
        calibrated_logits = cls.get_calibrated_logits(
            names, y_pos_score_list, missing, calibrators
        )
        weights = cls.fit_weights(names, y_true, calibrated_logits, missing)
        artifact = cls(
            fusion_schema_version=_FUSION_SCHEMA_VERSION,
            fusion_policy=_FUSION_POLICY,
            weight_objective=_WEIGHT_OBJECTIVE,
            names=names,
            labels=labels,
            positive_label=config.positive_label,
            weights=weights,
            calibrators=calibrators,
            thresholds={},
            default_threshold_name=get_default_threshold_name(config),
        )
        y_pos_score = artifact.predict_pos_score(names, y_pos_score_list, y_mask_list)
        available_any = ~missing.all(axis=1)
        thresholds = get_operating_thresholds(
            config,
            y_true[available_any],
            y_pos_score[available_any],
        )
        return cls(
            fusion_schema_version=artifact.fusion_schema_version,
            fusion_policy=artifact.fusion_policy,
            weight_objective=artifact.weight_objective,
            names=artifact.names,
            labels=artifact.labels,
            positive_label=artifact.positive_label,
            weights=artifact.weights,
            calibrators=artifact.calibrators,
            thresholds=thresholds,
            default_threshold_name=artifact.default_threshold_name,
        )

    @staticmethod
    def fit_calibrators(
        names: list[str],
        y_true: NDArray,
        y_pos_score_list: list[NDArray],
        y_mask_list: list[NDArray],
    ) -> dict[str, ModalityCalibrator]:
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
        return calibrators

    @staticmethod
    def get_calibrated_logits(
        names: list[str],
        y_pos_score_list: list[NDArray],
        missing: NDArray,
        calibrators: dict[str, ModalityCalibrator],
    ) -> NDArray:
        return np.column_stack(
            [
                calibrators[name].calibrated_logit(np.where(y_mask, 0.5, y_pos_score))
                for name, y_pos_score, y_mask in zip(
                    names, y_pos_score_list, missing.T, strict=True
                )
            ]
        )

    @classmethod
    def fit_weights(
        cls,
        names: list[str],
        y_true: NDArray,
        calibrated_logits: NDArray,
        missing: NDArray,
    ) -> dict[str, float]:
        if len(names) == 1:
            return {names[0]: 1.0}

        available_any = ~missing.all(axis=1)
        logits = calibrated_logits[available_any]
        masks = missing[available_any]
        target = y_true[available_any]
        initial = np.full(len(names), 1.0 / len(names))

        def objective(weights: NDArray) -> float:
            fused_logits = cls.fuse_logits(logits, masks, weights)
            return binary_log_loss_from_logits(target, fused_logits)

        result = minimize(
            objective,
            initial,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * len(names),
            constraints={'type': 'eq', 'fun': lambda weights: weights.sum() - 1.0},
            options={'ftol': 1e-9, 'maxiter': 1000},
        )
        if not result.success:
            raise IatreionException(
                'Failed to fit calibrated-fusion weights: $message',
                message=result.message,
            )
        weights = np.clip(result.x, 0.0, 1.0)
        weights /= weights.sum()
        return {
            name: weight.item() for name, weight in zip(names, weights, strict=True)
        }

    @staticmethod
    def fuse_logits(
        calibrated_logits: NDArray,
        missing: NDArray,
        weights: NDArray,
    ) -> NDArray:
        available = ~missing
        effective_weights = np.where(available, weights, 0.0)
        denominator = effective_weights.sum(axis=1)
        numerator = (calibrated_logits * effective_weights).sum(axis=1)

        fallback_denominator = available.sum(axis=1)
        fallback_numerator = (calibrated_logits * available).sum(axis=1)
        fallback = np.divide(
            fallback_numerator,
            fallback_denominator,
            out=np.zeros_like(fallback_numerator),
            where=fallback_denominator > 0,
        )
        return np.divide(
            numerator,
            denominator,
            out=fallback,
            where=denominator > 0,
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
        fused_logit = self.fuse_logits(calibrated_logits, missing, weights)
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
            return dict.fromkeys(names, 1 / len(names))
        return {name: self.weights[name] / total for name in names}

    @property
    def default_threshold(self) -> float:
        return self.thresholds[self.default_threshold_name]

    def predict_indices(
        self, y_pos_score: NDArray, *, threshold: float | None = None
    ) -> NDArray:
        threshold = self.default_threshold if threshold is None else threshold
        return (y_pos_score >= threshold).astype(int)

    def predict_labels(
        self, y_pos_score: NDArray, *, threshold: float | None = None
    ) -> NDArray:
        labels = np.asarray(self.labels)
        return labels[self.predict_indices(y_pos_score, threshold=threshold)]
