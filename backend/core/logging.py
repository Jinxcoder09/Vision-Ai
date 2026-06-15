"""
Eyeva AI - Structured logging via loguru.

Uses sys.stdout.reconfigure(encoding='utf-8') to prevent Windows CP1252
UnicodeEncodeError when loguru writes box-drawing or non-ASCII characters.
"""
import io
import sys
from loguru import logger


def _utf8_stdout() -> io.TextIOWrapper:
    """Return a UTF-8 reconfigured stdout wrapper, safe on Windows."""
    try:
        # Python 3.7+ — reconfigure the existing TextIOWrapper in-place
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        return sys.stdout
    except AttributeError:
        # Fallback: wrap stdout with UTF-8 writer
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def setup_logging(level: str = "INFO") -> None:
    _utf8_stdout()  # ensure stdout accepts unicode before adding sink
    logger.remove()

    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        "logs/eyeva.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    )

    logger.info("Logging configured - level={}", level)
