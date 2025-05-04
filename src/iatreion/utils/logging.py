import logging
from logging import FileHandler, Formatter, Logger, getLogger
from pathlib import Path
from types import MethodType

from rich.logging import RichHandler


def get_custom_logger(name: str | None = None) -> Logger:
    def _new_log(logger: Logger, level, msg, args, **kw) -> None:
        if isinstance(msg, str):
            for sub_msg in msg.split('\n'):
                logger._original_log(level, sub_msg, args, **kw)
        else:
            logger._original_log(level, msg, args, **kw)

    logger = getLogger(name)
    logger._original_log = MethodType(Logger._log, logger)
    logger._log = MethodType(_new_log, logger)
    return logger


logger = get_custom_logger('iatreion')
logger.setLevel(logging.DEBUG)
logger.addHandler(RichHandler(logging.INFO))


def add_file_handler(filename: Path, mode: str = 'w') -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    file_handler = FileHandler(filename, mode=mode)
    file_handler.setFormatter(Formatter('[%(asctime)s][%(levelname)8s] - %(message)s'))
    logger.addHandler(file_handler)
