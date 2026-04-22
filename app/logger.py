import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcko.log")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


def _setup_logger() -> logging.Logger:
    root_logger = logging.getLogger("mcko")
    root_logger.setLevel(LOG_LEVEL)

    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=2,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)

    root_logger.addHandler(file_handler)
    # Avoid duplicate lines when watchdog redirects stdout/stderr to the same log file.
    if not os.environ.get("MCKO_DISABLE_STDERR_LOGGING") and sys.stderr.isatty():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(LOG_LEVEL)
        root_logger.addHandler(stream_handler)

    root_logger.info("Logger initialized, log file: %s", LOG_FILE)
    return root_logger


logger = _setup_logger()
