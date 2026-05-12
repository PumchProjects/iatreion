import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
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

_console_min_level = ContextVar[int | None]('console_min_level', default=None)
_progress_enabled = ContextVar('progress_enabled', default=True)


class ConsoleLevelFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        min_level = _console_min_level.get()
        return min_level is None or record.levelno >= min_level


rich_handler = RichHandler(logging.INFO)
rich_handler.addFilter(ConsoleLevelFilter())
logger.addHandler(rich_handler)


def add_file_handler(filename: Path, *, format: bool = True) -> FileHandler:
    filename.parent.mkdir(parents=True, exist_ok=True)
    file_handler = FileHandler(filename, mode='w', encoding='utf-8')
    if format:
        file_handler.setFormatter(Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    logger.addHandler(file_handler)
    return file_handler


def remove_file_handler(file_handler: FileHandler) -> None:
    logger.removeHandler(file_handler)
    file_handler.close()


@contextmanager
def suppress_console_logs(
    min_level: int = logging.WARNING,
) -> Generator[None, None, None]:
    token = _console_min_level.set(min_level)
    try:
        yield
    finally:
        _console_min_level.reset(token)


@contextmanager
def disable_progress() -> Generator[None, None, None]:
    token = _progress_enabled.set(False)
    try:
        yield
    finally:
        _progress_enabled.reset(token)


progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    MofNCompleteColumn(),
)


@contextmanager
def task(description: str, total: int) -> Generator[Callable[[], None], None, None]:
    if total <= 1 or not _progress_enabled.get():
        yield lambda: None
        return
    task_id = progress.add_task(description, total=total)
    try:
        yield lambda: progress.update(task_id, advance=1)
    finally:
        progress.remove_task(task_id)
