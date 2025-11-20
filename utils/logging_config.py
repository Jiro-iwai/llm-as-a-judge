"""
Unified logging configuration module for LLM-as-a-Judge project.

This module provides a centralized logging system that replaces print() and
sys.stderr usage across all scripts. It supports:
- Log level control via environment variable (LOG_LEVEL)
- Consistent formatting with emojis and indentation
- File and console output support
- Backward compatibility with existing log functions
"""

import logging
import os
import sys
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def reset_logger() -> None:
    """Reset the global logger (mainly for testing purposes)."""
    global _logger
    _logger = None


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    reset: bool = False,
) -> logging.Logger:
    """
    Set up the logging system with specified configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from LOG_LEVEL environment variable or defaults to INFO.
        log_file: Optional file path to write logs to. If None, logs only to stderr.
        format_string: Optional custom format string for log messages.
                       If None, uses a default format.
        reset: If True, reset the logger even if it already exists (useful for testing).

    Returns:
        Configured logger instance.
    """
    global _logger

    # Determine log level
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(level_str, logging.INFO)

    # Create or reset logger
    if _logger is None or reset:
        _logger = logging.getLogger("llm_judge")
        _logger.setLevel(level)
        _logger.handlers.clear()  # Remove any existing handlers

        # Default format: simple message without timestamp/level for backward compatibility
        if format_string is None:
            format_string = "%(message)s"

        formatter = logging.Formatter(format_string)

        # Console handler (stderr)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)

    return _logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance, creating it if necessary.

    Returns:
        Logger instance.
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def log_info(message: str, indent: int = 0) -> None:
    """
    Log an info message with optional indentation.

    Args:
        message: Message to log.
        indent: Number of indentation levels (each level = 2 spaces).
    """
    logger = get_logger()
    prefix = "  " * indent
    logger.info(f"{prefix}{message}")


def log_section(title: str) -> None:
    """
    Log a section header with separator lines.

    Args:
        title: Section title.
    """
    logger = get_logger()
    separator = "=" * 70
    logger.info(f"\n{separator}")
    logger.info(f"{title}")
    logger.info(f"{separator}")


def log_warning(message: str, indent: int = 0) -> None:
    """
    Log a warning message with optional indentation.

    Args:
        message: Warning message to log.
        indent: Number of indentation levels (each level = 2 spaces).
    """
    logger = get_logger()
    prefix = "  " * indent
    logger.warning(f"{prefix}⚠️  {message}")


def log_error(message: str, indent: int = 0) -> None:
    """
    Log an error message with optional indentation.

    Args:
        message: Error message to log.
        indent: Number of indentation levels (each level = 2 spaces).
    """
    logger = get_logger()
    prefix = "  " * indent
    logger.error(f"{prefix}❌ {message}")


def log_success(message: str, indent: int = 0) -> None:
    """
    Log a success message with optional indentation.

    Args:
        message: Success message to log.
        indent: Number of indentation levels (each level = 2 spaces).
    """
    logger = get_logger()
    prefix = "  " * indent
    logger.info(f"{prefix}✓ {message}")

