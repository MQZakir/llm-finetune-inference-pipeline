"""
Structured logging with Rich formatting and optional JSON output for production.

Usage
-----
>>> from src.utils.logging import get_logger, setup_logging
>>> setup_logging(level="INFO", json_output=False)
>>> logger = get_logger(__name__)
>>> logger.info("Training started", extra={"step": 0, "lr": 2e-4})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any

# Try to use Rich for pretty console output
try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """
    Emit log records as newline-delimited JSON.
    Useful for log aggregators (Datadog, Loki, CloudWatch).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via `extra=`
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure the root logger for the project.

    Parameters
    ----------
    level : str
        Logging level — DEBUG, INFO, WARNING, ERROR, CRITICAL.
    json_output : bool
        If True, emit JSON lines to stdout (for production log pipelines).
        If False, use Rich formatting for human-readable output.
    log_file : str | None
        If provided, also write plain-text logs to this path.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers
    root.handlers.clear()

    if json_output:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    elif _RICH_AVAILABLE:
        handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("transformers", "datasets", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Call setup_logging() once at programme start."""
    return logging.getLogger(name)
