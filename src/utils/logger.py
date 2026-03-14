"""Loguru-based logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


_configured = False


def setup_logger(log_level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure loguru with console + file rotation."""
    global _configured
    if _configured:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path / "aiquant_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        )

    _configured = True


def get_logger(name: str = "aiquant") -> logger.__class__:
    """Get a contextualized logger."""
    return logger.bind(name=name)


def reset_logger() -> None:
    """Reset logger state (for testing)."""
    global _configured
    _configured = False
    logger.remove()
