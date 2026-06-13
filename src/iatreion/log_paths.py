from pathlib import Path

TRAINING_DIR = 'training'
FINAL_DIR = 'final'
FINAL_CALIBRATION_DIR = 'final-calibration'
OPTUNA_DIR = 'optuna'
EVAL_DIR = 'eval'
RESULT_REPLAY_DIR = 'result-replay'
NESTED_DIR = 'nested'


def training_root(log_root: Path, dataset_name: str, group_name: str) -> Path:
    return log_root / TRAINING_DIR / dataset_name / group_name


def training_model_root(
    log_root: Path,
    dataset_name: str,
    group_name: str,
    model_name: str,
    aggregate: str,
) -> Path:
    return training_root(log_root, dataset_name, group_name) / model_name / aggregate


def final_root(log_root: Path, group_name: str) -> Path:
    return log_root / FINAL_DIR / group_name


def final_model_root(log_root: Path, group_name: str, model_name: str) -> Path:
    return final_root(log_root, group_name) / model_name


def final_calibration_root(
    log_root: Path,
    dataset_name: str,
    group_name: str,
) -> Path:
    return log_root / FINAL_CALIBRATION_DIR / dataset_name / group_name


def final_calibration_model_root(
    log_root: Path,
    dataset_name: str,
    group_name: str,
    model_name: str,
    aggregate: str,
) -> Path:
    return (
        final_calibration_root(log_root, dataset_name, group_name)
        / model_name
        / aggregate
    )


def optuna_root(log_root: Path) -> Path:
    return log_root / OPTUNA_DIR
