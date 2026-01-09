"""Structured logging configuration for Refactron.

This module provides JSON-formatted logging for CI/CD and log aggregation systems,
with configurable log levels and rotation support.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with JSON formatting and rotation support."""

    def __init__(
        self,
        name: str = "refactron",
        level: str = "INFO",
        log_file: Optional[Path] = None,
        log_format: str = "json",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """Initialize structured logger.

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (if None, uses default location)
            log_format: Log format ('json' or 'text')
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
        """
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.log_format = log_format
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file

        # Set default log file location if not provided
        if log_file is None and enable_file:
            log_dir = Path.home() / ".refactron" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = log_dir / "refactron.log"
        else:
            self.log_file = log_file

        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(self.level)

        # Configure formatter
        if self.log_format == "json":
            formatter: logging.Formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add rotating file handler
        if self.enable_file and self.log_file:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance.

        Returns:
            Configured logger instance
        """
        return self.logger

    def log_with_context(
        self, level: str, message: str, extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log message with additional context data.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            extra_data: Additional context data to include in log
        """
        log_func = getattr(self.logger, level.lower())
        if extra_data:
            # Create a LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name,
                getattr(logging, level.upper()),
                "(unknown file)",
                0,
                message,
                (),
                None,
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            log_func(message)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "json",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True,
) -> StructuredLogger:
    """Setup structured logging for Refactron.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log format ('json' or 'text')
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        enable_console: Enable console logging
        enable_file: Enable file logging

    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(
        name="refactron",
        level=level,
        log_file=log_file,
        log_format=log_format,
        max_bytes=max_bytes,
        backup_count=backup_count,
        enable_console=enable_console,
        enable_file=enable_file,
    )
