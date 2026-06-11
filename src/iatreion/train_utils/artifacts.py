from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .splitter import TrainStepContext

ARTIFACT_ROOT = 'artifacts'
TRANSFORM_ARTIFACT_FILE = 'transform.toml'

RRL_RULE_FILE = 'rules.tsv'
RRL_FEATURE_SELECTION_FILE = 'feature-selection.toml'
RRL_SIMPLE_IMPUTER_FILE = 'simple-imputer.toml'

RRL_FOLD_FEATURE_SELECTION_SUFFIX = '.feature-selection.toml'
RRL_FOLD_SIMPLE_IMPUTER_SUFFIX = '.simple-imputer.toml'


def get_artifact_dir(root: Path, name: str) -> Path:
    return root / ARTIFACT_ROOT / name


def get_transform_artifact_path(root: Path, name: str) -> Path:
    return get_artifact_dir(root, name) / TRANSFORM_ARTIFACT_FILE


def get_final_rrl_rule_path(root: Path, name: str) -> Path:
    return get_artifact_dir(root, name) / RRL_RULE_FILE


def get_rrl_rule_path(root: Path, ctx: TrainStepContext) -> Path:
    if ctx.db_enc.train.final:
        return get_final_rrl_rule_path(root, ctx.name)
    return root / ctx.rrl_file


def get_rrl_feature_selection_path(rule_path: Path) -> Path:
    if rule_path.name == RRL_RULE_FILE:
        return rule_path.with_name(RRL_FEATURE_SELECTION_FILE)
    return rule_path.with_name(f'{rule_path.stem}{RRL_FOLD_FEATURE_SELECTION_SUFFIX}')


def get_rrl_simple_imputer_path(rule_path: Path) -> Path:
    if rule_path.name == RRL_RULE_FILE:
        return rule_path.with_name(RRL_SIMPLE_IMPUTER_FILE)
    return rule_path.with_name(f'{rule_path.stem}{RRL_FOLD_SIMPLE_IMPUTER_SUFFIX}')
