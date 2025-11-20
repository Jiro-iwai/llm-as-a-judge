"""
Unit tests for utils.logging_config module.

This module tests the unified logging system that replaces print() and sys.stderr
usage across all scripts.
"""

import io
import logging
import os
from unittest.mock import patch


# Import the logging module (will be created)
# Note: This will fail until we create utils/logging_config.py
from utils.logging_config import (
    get_logger,
    log_info,
    log_error,
    log_warning,
    log_success,
    log_section,
    setup_logging,
)


def reset_logger():
    """Reset the global logger for testing"""
    from utils.logging_config import reset_logger as reset
    reset()


class TestSetupLogging:
    """Tests for setup_logging function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_setup_logging_default(self):
        """Test setup_logging with default settings"""
        logger = setup_logging(reset=True)
        assert logger is not None
        assert logger.level == logging.INFO

    def test_setup_logging_with_level(self):
        """Test setup_logging with custom log level"""
        logger = setup_logging(level=logging.DEBUG, reset=True)
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_env_var(self):
        """Test setup_logging reads LOG_LEVEL from environment"""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            logger = setup_logging(reset=True)
            assert logger.level == logging.DEBUG

    def test_setup_logging_with_invalid_env_var(self):
        """Test setup_logging handles invalid LOG_LEVEL gracefully"""
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            logger = setup_logging(reset=True)
            # Should default to INFO
            assert logger.level == logging.INFO


class TestGetLogger:
    """Tests for get_logger function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance"""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_instance(self):
        """Test get_logger returns the same logger instance"""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2


class TestLogInfo:
    """Tests for log_info function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_log_info_basic(self):
        """Test log_info outputs message"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_info("Test message")
        output = stream.getvalue()
        assert "Test message" in output

    def test_log_info_with_indent(self):
        """Test log_info with indentation"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_info("Test message", indent=2)
        output = stream.getvalue()
        assert "Test message" in output
        assert output.startswith("    ")  # 2 indents = 4 spaces

    def test_log_info_log_level_filtering(self):
        """Test log_info respects log level"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(level=logging.ERROR, reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        log_info("Test message")
        output = stream.getvalue()
        # Should not output at ERROR level
        assert "Test message" not in output


class TestLogSection:
    """Tests for log_section function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_log_section_output(self):
        """Test log_section outputs formatted section header"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_section("Test Section")
        output = stream.getvalue()
        assert "Test Section" in output
        assert "=" * 70 in output

    def test_log_section_format(self):
        """Test log_section has correct format"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_section("Test Section")
        output = stream.getvalue()
        lines = [line for line in output.strip().split("\n") if line.strip()]
        assert len(lines) == 3  # Separator, title, separator
        assert lines[0] == "=" * 70
        assert lines[1] == "Test Section"
        assert lines[2] == "=" * 70


class TestLogWarning:
    """Tests for log_warning function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_log_warning_basic(self):
        """Test log_warning outputs warning message"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        log_warning("Warning message")
        output = stream.getvalue()
        assert "Warning message" in output
        assert "⚠️" in output

    def test_log_warning_with_indent(self):
        """Test log_warning with indentation"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        log_warning("Warning message", indent=1)
        output = stream.getvalue()
        assert "Warning message" in output
        assert output.startswith("  ")  # 1 indent = 2 spaces


class TestLogError:
    """Tests for log_error function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_log_error_basic(self):
        """Test log_error outputs error message"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        log_error("Error message")
        output = stream.getvalue()
        assert "Error message" in output
        assert "❌" in output

    def test_log_error_with_indent(self):
        """Test log_error with indentation"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        log_error("Error message", indent=1)
        output = stream.getvalue()
        assert "Error message" in output
        assert output.startswith("  ")  # 1 indent = 2 spaces


class TestLogSuccess:
    """Tests for log_success function"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_log_success_basic(self):
        """Test log_success outputs success message"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_success("Success message")
        output = stream.getvalue()
        assert "Success message" in output
        assert "✓" in output

    def test_log_success_with_indent(self):
        """Test log_success with indentation"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_success("Success message", indent=1)
        output = stream.getvalue()
        assert "Success message" in output
        assert output.startswith("  ")  # 1 indent = 2 spaces


class TestLoggingIntegration:
    """Integration tests for logging system"""

    def setup_method(self):
        """Reset logger before each test"""
        reset_logger()

    def test_all_log_functions_work_together(self):
        """Test all log functions work together"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_info("Info message")
        log_warning("Warning message")
        log_error("Error message")
        log_success("Success message")
        log_section("Section Title")

        output = stream.getvalue()
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output
        assert "Success message" in output
        assert "Section Title" in output

    def test_logging_with_file_handler(self):
        """Test logging can output to file"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_file = f.name

        try:
            logger = setup_logging(level=logging.INFO, log_file=temp_file, reset=True)
            log_info("Test message to file")
            
            # Flush handlers to ensure content is written
            for handler in logger.handlers:
                handler.flush()

            # Read the file
            with open(temp_file, "r") as f:
                content = f.read()
                assert "Test message to file" in content
        finally:
            os.unlink(temp_file)

    def test_logging_levels(self):
        """Test different log levels work correctly"""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(level=logging.WARNING, reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        log_info("Info message")  # Should not appear
        log_warning("Warning message")  # Should appear
        log_error("Error message")  # Should appear

        output = stream.getvalue()
        assert "Info message" not in output
        assert "Warning message" in output
        assert "Error message" in output

