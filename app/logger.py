import os
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Custom formatter with millisecond precision and clean module tags."""
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            s = f"{t}.{int(record.msecs):03d}"
        return s


def setup_logger(
    name: str = "FaceTracker",
    log_file: str = "logs/events.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB rotating threshold
    backup_count: int = 5
) -> logging.Logger:
    """
    Configures and returns a thread-safe logger with:
    1. RotatingFileHandler for persistent auditing in events.log.
    2. StreamHandler for real-time console feedback.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already initialized
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 1. Rotating File Handler
    file_fmt = CustomFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # 2. Console Stream Handler
    console_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "FaceTracker") -> logging.Logger:
    """Retrieves an existing configured logger or initializes with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
