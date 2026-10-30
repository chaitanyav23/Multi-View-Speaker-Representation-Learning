"""
scripts/utils/logger.py
=======================
Centralised logging factory for the BWC pipeline.

Provides:
  - Rich-enhanced coloured console output
  - Rotating file handler
  - A single get_logger() entry-point used by all modules
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# rich is optional — gracefully degrade if not installed
try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ── Module-level registry so we never create duplicate handlers ──────────────
_INITIALISED: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "bwc_pipeline",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_filename: str = "stage1_pipeline.log",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
    rich_console: bool = True,
) -> logging.Logger:
    """
    Return a configured :class:`logging.Logger`.

    Parameters
    ----------
    name:          Logger name (use __name__ in calling modules).
    level:         String log level — "DEBUG" | "INFO" | "WARNING" | "ERROR".
    log_dir:       Directory for the rotating file handler.  ``None`` disables
                   file logging.
    log_filename:  Filename inside *log_dir*.
    max_bytes:     Max single log file size before rotation.
    backup_count:  Number of rotated backups to keep.
    rich_console:  Use ``rich.logging.RichHandler`` for pretty console output.

    Returns
    -------
    logging.Logger
    """
    if name in _INITIALISED:
        return _INITIALISED[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False          # don't bubble up to root logger

    fmt_file = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────────
    if rich_console and _RICH_AVAILABLE:
        console_handler = RichHandler(
            level=getattr(logging, level.upper(), logging.INFO),
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(fmt_file)

    logger.addHandler(console_handler)

    # ── Rotating file handler ────────────────────────────────────────────────
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)   # always write everything to disk
        file_handler.setFormatter(fmt_file)
        logger.addHandler(file_handler)

    _INITIALISED[name] = logger
    return logger


def get_child_logger(parent_name: str, child_name: str) -> logging.Logger:
    """Return a child logger inheriting parent's handlers."""
    return logging.getLogger(f"{parent_name}.{child_name}")
