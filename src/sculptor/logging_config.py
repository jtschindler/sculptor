"""Logging for sculptor and sculptor_extensions.

Every module logs through `logging.getLogger(__name__)`, i.e. a child of the
`sculptor` or `sculptor_extensions` logger. Importing sculptor calls
`configure_logging()` once with its defaults, so INFO messages are shown
like the `print` output they replace:

- messages below WARNING go to stdout, WARNING and above to stderr;
- only the `sculptor` / `sculptor_extensions` loggers are configured: the
  root logger and other libraries' loggers are left alone.

Change the defaults by calling `configure_logging()` again (it replaces its
own handlers, never adds duplicates), or with environment variables read
when sculptor is imported:

- `SCULPTOR_LOG_LEVEL`: e.g. `WARNING` to hide progress messages;
- `SCULPTOR_LOG_TO_FILE=1`: also write `~/.sculptor/logs/sculptor.log`
  (an existing log is kept as `sculptor_<timestamp>.log.bak`).

To send sculptor's messages through your own logging setup instead (e.g.
handlers on the root logger), call `reset_logging()`.

Based on the `configure_logging` helper of the `euclid_qso_selection`
package.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime as dt
from functools import lru_cache
from pathlib import Path

PACKAGE_LOGGERS = ("sculptor", "sculptor_extensions")
DEFAULT_LEVEL = logging.INFO  # level of the package loggers
FLT_LEVEL = logging.WARNING  # min level that goes to stderr

LOG_LEVEL_ENV_VAR = "SCULPTOR_LOG_LEVEL"
LOG_TO_FILE_ENV_VAR = "SCULPTOR_LOG_TO_FILE"
DEFAULT_LOCAL_PATH = Path.home() / ".sculptor"

# add this for the date as well: `%(asctime)s - `
LOGGING_FORMATTER_STR = "%(name)s - %(levelname)s: %(message)s"

# Marks the handlers this module installed, so they can be replaced or
# removed without touching handlers added by anyone else.
_OWN_HANDLER_ATTR = "_sculptor_handler"


class LogFilter(logging.Filter):
    """Pass only records below `level` (the stdout side of the split)."""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


class _StdStreamHandler(logging.StreamHandler):
    """StreamHandler that looks up `sys.stdout`/`sys.stderr` at every write.

    A plain `StreamHandler(sys.stdout)` keeps the stream object it was given
    at import time, so output redirected later (Jupyter, pytest's capture,
    `contextlib.redirect_stdout`) would bypass the redirection.
    """

    def __init__(self, stream_name):
        self._stream_name = stream_name
        super().__init__(getattr(sys, stream_name))

    @property
    def stream(self):
        return getattr(sys, self._stream_name)

    @stream.setter
    def stream(self, value):
        # StreamHandler.__init__ assigns the stream; always use the live one.
        pass


def _parse_level(value, source="level"):
    """Return a numeric logging level from an int or a name like "info"."""

    if isinstance(value, int):
        return value
    levels = logging.getLevelNamesMapping()
    name = str(value).strip().upper()
    if name not in levels:
        raise ValueError(
            f"Unknown {source}={value!r}; use e.g. DEBUG, INFO, WARNING or ERROR."
        )
    return levels[name]


def _level_from_env(default):
    value = os.environ.get(LOG_LEVEL_ENV_VAR)
    if not value:
        return default
    return _parse_level(value, source=LOG_LEVEL_ENV_VAR)


def _own(handler):
    setattr(handler, _OWN_HANDLER_ATTR, True)
    return handler


def _remove_own_handlers(logger):
    for handler in list(logger.handlers):
        if getattr(handler, _OWN_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    *,
    level=None,
    flt_level=FLT_LEVEL,
    log_to_file=None,
    local_path=None,
    capture_warnings=False,
    silence=None,
):
    """Configure the sculptor loggers and return the `sculptor` logger.

    Safe to call repeatedly: each call replaces the handlers installed by
    the previous one.

    :param level: Level of the package loggers. Defaults to the
        `SCULPTOR_LOG_LEVEL` environment variable, else INFO.
    :type level: int or str
    :param flt_level: Messages below this level go to stdout, the rest to
        stderr (default: WARNING).
    :type flt_level: int
    :param log_to_file: Also write a log file. Defaults to the
        `SCULPTOR_LOG_TO_FILE` environment variable ("1"/"true").
    :type log_to_file: bool
    :param local_path: Base directory for the log file, written to
        `<local_path>/logs/sculptor.log` (default: `~/.sculptor`). An
        existing log is renamed to `sculptor_<timestamp>.log.bak`.
    :type local_path: pathlib.Path
    :param capture_warnings: Route Python `warnings` through the same
        handlers (logger `py.warnings`).
    :type capture_warnings: bool
    :param silence: Names of other libraries' loggers to set to WARNING.
    :type silence: list(str)
    :return: The `sculptor` logger
    :rtype: logging.Logger
    """

    level = _level_from_env(DEFAULT_LEVEL) if level is None else _parse_level(level)

    formatter = logging.Formatter(fmt=LOGGING_FORMATTER_STR)

    # stdout handler (< flt_level)
    i_handler = _own(_StdStreamHandler("stdout"))
    i_handler.addFilter(LogFilter(flt_level))
    i_handler.setFormatter(formatter)

    # stderr handler (>= flt_level)
    e_handler = _own(_StdStreamHandler("stderr"))
    e_handler.setLevel(flt_level)
    e_handler.setFormatter(formatter)

    handlers = [i_handler, e_handler]

    if log_to_file is None:
        log_to_file = os.getenv(LOG_TO_FILE_ENV_VAR, "") in {"1", "true", "True"}

    if log_to_file:
        log_dir = Path(local_path or DEFAULT_LOCAL_PATH) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "sculptor.log"
        if log_file.exists():
            # local time of the old log's last write
            mtime = dt.fromtimestamp(log_file.stat().st_mtime, tz=UTC).astimezone()
            ts = mtime.strftime("%Y-%m-%dT%H-%M-%S")
            log_file.rename(log_dir / f"sculptor_{ts}.log.bak")

        f_handler = _own(logging.FileHandler(log_file))
        f_handler.setFormatter(formatter)
        handlers.append(f_handler)

    names = list(PACKAGE_LOGGERS)
    if capture_warnings:
        logging.captureWarnings(True)
        names.append("py.warnings")

    for name in names:
        logger = logging.getLogger(name)
        _remove_own_handlers(logger)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        # Our handlers print the message; don't print it again via root.
        logger.propagate = False

    for name in silence or []:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("sculptor")


def reset_logging():
    """Remove sculptor's handlers and let its messages propagate to the root
    logger, so an application's own logging setup handles them."""

    for name in (*PACKAGE_LOGGERS, "py.warnings"):
        logger = logging.getLogger(name)
        _remove_own_handlers(logger)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


# In case I need to suppress things temporarily
# thanks to: https://gist.github.com/simon-weber/7853144
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """Context manager that drops every logging message issued in its body.

    :param highest_level: The maximum logging level in use. Only needs
        changing if a custom level above CRITICAL is defined.
    """

    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


@lru_cache(None)
def warn_once(logger, msg):
    """Log `msg` as a warning only the first time it is seen for `logger`."""

    logger.warning(msg)
