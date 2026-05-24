"""Logging bootstrap utilities for the application."""

import logging
from pathlib import Path


def configure_logging(log_level: str, log_format: str, logs_dir: Path) -> None:
    """Configure root logger with console and file handlers once."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
