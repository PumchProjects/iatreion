import os
from pathlib import Path


def get_config_path() -> Path:
    if (config_path := os.getenv('IATREION_CONFIG_PATH')) is not None:
        return Path(config_path)
    if (pyfuze_path := os.getenv('PYFUZE_EXECUTABLE_PATH')) is not None:
        return Path(pyfuze_path).parent / 'config.toml'
    return Path('config.toml')
