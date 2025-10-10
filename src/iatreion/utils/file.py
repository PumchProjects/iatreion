import os
from pathlib import Path
from typing import Any

import tomli
import tomli_w


def get_config_path() -> Path:
    if (config_path := os.getenv('IATREION_CONFIG_PATH')) is not None:
        return Path(config_path)
    if (pyfuze_path := os.getenv('PYFUZE_EXECUTABLE_PATH')) is not None:
        return Path(pyfuze_path).parent / 'config.toml'
    return Path('config.toml')


def load_dict(path: Path) -> dict[str, Any]:
    config_dict: dict[str, Any] = {}
    if path.is_file():
        with path.open('rb') as f:
            config_dict = tomli.load(f)
    return config_dict


def order(d: dict, /) -> dict:
    return {k: order(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}


def save_dict(config_dict: dict[str, Any], path: Path) -> None:
    with path.open('wb') as f:
        tomli_w.dump(order(config_dict), f)
