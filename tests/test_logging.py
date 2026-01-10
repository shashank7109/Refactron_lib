"""Tests for structured logging configuration."""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from refactron.core.logging_config import JSONFormatter, StructuredLogger, setup_logging


class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_format_basic_log(self):
        """Test basic log record formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 10
        assert "timestamp" in log_data

    def test_format_with_exception(self):
        """Test log record with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "ERROR"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert "Test error" in log_data["exception"]["message"]

    def test_format_with_extra_data(self):
        """Test log record with extra data."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=30,
            msg="Test with context",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"file_path": "/path/to/file.py", "duration_ms": 100.5}

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["file_path"] == "/path/to/file.py"
        assert log_data["duration_ms"] == 100.5


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_initialization_defaults(self):
        """Test logger initialization with defaults."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = StructuredLogger(
                name="test",
                level="INFO",
                log_file=log_file,
                enable_console=False,
                enable_file=True,
            )

            assert logger.name == "test"
            assert logger.level == logging.INFO
            assert logger.log_file == log_file

    def test_json_logging(self):
        """Test JSON format logging to file."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = StructuredLogger(
                name="test",
                level="INFO",
                log_file=log_file,
                log_format="json",
                enable_console=False,
                enable_file=True,
            )

            logger.get_logger().info("Test message")

            # Read and verify log file
            with open(log_file, "r") as f:
                log_line = f.read().strip()
                log_data = json.loads(log_line)

            assert log_data["message"] == "Test message"
            assert log_data["level"] == "INFO"

    def test_text_logging(self):
        """Test text format logging to file."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = StructuredLogger(
                name="test",
                level="INFO",
                log_file=log_file,
                log_format="text",
                enable_console=False,
                enable_file=True,
            )

            logger.get_logger().info("Test text message")

            # Read and verify log file
            with open(log_file, "r") as f:
                log_line = f.read().strip()

            assert "Test text message" in log_line
            assert "INFO" in log_line

    def test_log_rotation(self):
        """Test that log rotation configuration is applied."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            max_bytes = 1024
            backup_count = 3

            logger = StructuredLogger(
                name="test",
                level="INFO",
                log_file=log_file,
                max_bytes=max_bytes,
                backup_count=backup_count,
                enable_console=False,
                enable_file=True,
            )

            # Verify handler is configured with rotation
            file_handler = None
            for handler in logger.get_logger().handlers:
                if hasattr(handler, "maxBytes"):
                    file_handler = handler
                    break

            assert file_handler is not None
            assert file_handler.maxBytes == max_bytes
            assert file_handler.backupCount == backup_count

    def test_log_levels(self):
        """Test different log levels."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Test with WARNING level
            logger = StructuredLogger(
                name="test",
                level="WARNING",
                log_file=log_file,
                log_format="json",
                enable_console=False,
                enable_file=True,
            )

            logger.get_logger().info("Info message")
            logger.get_logger().warning("Warning message")
            logger.get_logger().error("Error message")

            # Read log file
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Only WARNING and ERROR should be logged
            assert len(lines) == 2

            log_data_1 = json.loads(lines[0])
            log_data_2 = json.loads(lines[1])

            assert log_data_1["level"] == "WARNING"
            assert log_data_2["level"] == "ERROR"

    def test_disable_console_logging(self):
        """Test disabling console logging."""
        logger = StructuredLogger(
            name="test",
            level="INFO",
            enable_console=False,
            enable_file=False,
        )

        # Verify no console handlers
        console_handlers = [
            h
            for h in logger.get_logger().handlers
            if isinstance(h, logging.StreamHandler) and h.stream.name == "<stdout>"
        ]
        assert len(console_handlers) == 0

    def test_disable_file_logging(self):
        """Test disabling file logging."""
        logger = StructuredLogger(
            name="test",
            level="INFO",
            enable_console=True,
            enable_file=False,
        )

        # Verify no file handlers
        file_handlers = [h for h in logger.get_logger().handlers if hasattr(h, "baseFilename")]
        assert len(file_handlers) == 0


def test_setup_logging():
    """Test setup_logging convenience function."""
    with TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logging(
            level="DEBUG",
            log_file=log_file,
            log_format="json",
            enable_console=False,
            enable_file=True,
        )

        assert isinstance(logger, StructuredLogger)
        assert logger.level == logging.DEBUG
        assert logger.log_file == log_file
