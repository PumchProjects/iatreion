import hashlib
import math
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from iatreion.configs import ModelConfig
from iatreion.utils import load_dict, save_dict

MANIFEST_FILE = 'manifest.toml'
SKIPPED_CONFIG_FIELDS = {'_log_handler'}


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rel_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def run_text(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.rstrip('\r\n')


def file_record(path: Path, *, base: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        'path': rel_path(path, base),
        'size': stat.st_size,
        'sha256': sha256_file(path),
    }


def git_info() -> dict[str, Any]:
    root = Path(run_text('git', 'rev-parse', '--show-toplevel') or '.')
    dirty_files = [
        line[3:] for line in run_text('git', 'status', '--short', cwd=root).splitlines()
    ]
    return {
        'root': str(root),
        'branch': run_text('git', 'branch', '--show-current', cwd=root),
        'commit': run_text('git', 'rev-parse', 'HEAD', cwd=root),
        'dirty': bool(dirty_files),
        'dirty_files': dirty_files,
    }


def lock_info(repo_root: Path) -> dict[str, Any]:
    path = repo_root / 'uv.lock'
    data = load_dict(path)
    return {
        'path': str(path),
        'version': data.get('version', ''),
        'revision': data.get('revision', ''),
        'sha256': sha256_file(path),
    }


def process_command() -> dict[str, Any]:
    data = {
        'argv': shlex.join(sys.argv),
        'python_argv': shlex.join(getattr(sys, 'orig_argv', sys.argv)),
    }
    proc_cmdline = Path('/proc/self/cmdline')
    if proc_cmdline.is_file():
        parts = proc_cmdline.read_bytes().rstrip(b'\0').split(b'\0')
        data['process'] = shlex.join(part.decode() for part in parts)
    return data


def torch_info() -> dict[str, Any]:
    import torch

    data: dict[str, Any] = {
        'version': torch.__version__,
        'cuda_build': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count(),
        'cudnn_version': torch.backends.cudnn.version(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', ''),
    }
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        data['allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
    data['gpus'] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            data['gpus'].append(
                {
                    'index': index,
                    'name': props.name,
                    'total_memory_mb': props.total_memory // 1024**2,
                    'capability': f'{props.major}.{props.minor}',
                }
            )
    return data


def environment_info() -> dict[str, Any]:
    data = {
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'uv': run_text('uv', '--version'),
    }
    try:
        data['torch'] = torch_info()
    except ImportError:
        data['torch'] = {'installed': False}
    return data


def dataclass_dict(obj: Any) -> Any:
    if not is_dataclass(obj):
        return obj
    return {
        field.name: dataclass_dict(getattr(obj, field.name))
        for field in fields(obj)
        if field.name not in SKIPPED_CONFIG_FIELDS
    }


def toml_value(value: Any) -> Any:
    if value is None:
        return 'none'
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): toml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [toml_value(item) for item in value]
    if is_dataclass(value):
        return toml_value(dataclass_dict(value))
    return str(value)


def dataset_files(config: ModelConfig, repo_root: Path) -> list[dict[str, Any]]:
    dataset = config.dataset
    paths = [dataset.get_data(name) for name in dataset.names]
    paths += [dataset.get_info(name) for name in dataset.names]
    process_info = dataset.prefix / 'process_info.toml'
    if process_info.is_file():
        paths.append(process_info)
    return [file_record(path, base=repo_root) for path in sorted(set(paths))]


def artifact_files(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    paths = [
        path
        for path in root.rglob('*')
        if path.is_file() and path.name != MANIFEST_FILE
    ]
    return [file_record(path, base=root) for path in sorted(paths)]


def write_manifest(
    config: ModelConfig,
    *,
    started_at: str,
    objectives: Mapping[str, float],
    parameter_map: Mapping[Any, Any],
) -> Path:
    git = git_info()
    repo_root = Path(git['root'])
    root = config.train._log_dir
    manifest = {
        'run': {
            'started_at': started_at,
            'finished_at': now_utc(),
            'cwd': os.getcwd(),
            'model_config': type(config).__name__,
            'output_root': str(root),
            'tuned': config.tune,
        },
        'command': process_command(),
        'git': git,
        'lock': lock_info(repo_root),
        'environment': environment_info(),
        'dataset_files': dataset_files(config, repo_root),
        'hyperparameters': dataclass_dict(config),
        'selected_hyperparameters': parameter_map,
        'metrics': dict(objectives),
        'artifacts': artifact_files(root),
    }
    path = root / MANIFEST_FILE
    save_dict(toml_value(manifest), path)
    return path
