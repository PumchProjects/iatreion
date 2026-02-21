import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from logging import FileHandler, Formatter, Logger, getLogger
from pathlib import Path
from types import MethodType

from rich.logging import RichHandler
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn


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


def add_file_handler(filename: Path, mode: str = 'w') -> FileHandler:
    filename.parent.mkdir(parents=True, exist_ok=True)
    file_handler = FileHandler(filename, mode=mode)
    file_handler.setFormatter(Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)
    return file_handler


def remove_file_handler(file_handler: FileHandler) -> None:
    logger.removeHandler(file_handler)
    file_handler.close()


progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    MofNCompleteColumn(),
)


@contextmanager
def task(
    description: str, total: int, predicate: bool = True
) -> Generator[Callable[[], None], None, None]:
    if not predicate:
        yield lambda: None
        return
    task_id = progress.add_task(description, total=total)
    try:
        yield lambda: progress.update(task_id, advance=1)
    finally:
        progress.remove_task(task_id)
