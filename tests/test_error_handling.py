"""Tests for graceful degradation and error handling in Refactron core."""

import tempfile
from pathlib import Path

import pytest

from refactron import Refactron
from refactron.core.config import RefactronConfig
from refactron.core.exceptions import ConfigError


class TestGracefulDegradation:
    """Test graceful degradation when files fail analysis."""

    def test_analyze_continues_after_file_error(self) -> None:
        """Test that analysis continues when one file fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a valid Python file
            valid_file = tmpdir_path / "valid.py"
            valid_file.write_text("def valid_function():\n    return 42\n")

            # Create a file with invalid encoding
            invalid_file = tmpdir_path / "invalid.py"
            # Write binary data that will cause encoding error
            invalid_file.write_bytes(b"\xff\xfe\x00invalid python")

            # Create another valid file
            valid_file2 = tmpdir_path / "valid2.py"
            valid_file2.write_text("def another_valid():\n    return 'hello'\n")

            refactron = Refactron()
            result = refactron.analyze(tmpdir_path)

            # Should have analyzed 2 valid files and recorded 1 failure
            assert result.files_analyzed_successfully == 2
            assert result.files_failed == 1
            assert result.total_files == 3

    def test_failed_files_recorded_in_result(self) -> None:
        """Test that failed files are properly recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a file with invalid encoding
            invalid_file = tmpdir_path / "invalid.py"
            invalid_file.write_bytes(b"\xff\xfe\x00invalid")

            refactron = Refactron()
            result = refactron.analyze(tmpdir_path)

            assert len(result.failed_files) == 1
            failed = result.failed_files[0]
            assert failed.file_path == invalid_file
            assert failed.error_type == "AnalysisError"
            assert failed.recovery_suggestion is not None

    def test_analyze_single_invalid_file_raises_error(self) -> None:
        """Test that analyzing a single invalid file raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a file with invalid encoding
            invalid_file = tmpdir_path / "invalid.py"
            invalid_file.write_bytes(b"\xff\xfe\x00invalid")

            refactron = Refactron()

            # When analyzing a single file, errors should be captured
            result = refactron.analyze(invalid_file)

            # Should have 1 failed file
            assert result.files_failed == 1

    def test_summary_includes_failed_files_count(self) -> None:
        """Test that summary includes failed files count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            valid_file = tmpdir_path / "valid.py"
            valid_file.write_text("def valid():\n    pass\n")

            invalid_file = tmpdir_path / "invalid.py"
            invalid_file.write_bytes(b"\xff\xfe\x00invalid")

            refactron = Refactron()
            result = refactron.analyze(tmpdir_path)
            summary = result.summary()

            assert summary["total_files"] == 2
            assert summary["files_analyzed"] == 1
            assert summary["files_failed"] == 1

    def test_report_shows_failed_files(self) -> None:
        """Test that report includes failed files section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            invalid_file = tmpdir_path / "invalid.py"
            invalid_file.write_bytes(b"\xff\xfe\x00invalid")

            refactron = Refactron()
            result = refactron.analyze(tmpdir_path)
            report = result.report()

            assert "FAILED FILES" in report
            assert "invalid.py" in report
            assert "💡" in report  # Recovery suggestion indicator


class TestConfigErrorHandling:
    """Test configuration error handling."""

    def test_load_nonexistent_config_raises_error(self) -> None:
        """Test that loading non-existent config raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            RefactronConfig.from_file(Path("/nonexistent/config.yaml"))

        error = exc_info.value
        assert "not found" in str(error).lower()
        assert error.recovery_suggestion is not None

    def test_config_error_has_file_path(self) -> None:
        """Test that ConfigError includes file path."""
        config_path = Path("/nonexistent/config.yaml")
        with pytest.raises(ConfigError) as exc_info:
            RefactronConfig.from_file(config_path)

        error = exc_info.value
        assert error.file_path == config_path

    def test_invalid_yaml_raises_config_error(self) -> None:
        """Test that invalid YAML raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write invalid YAML
            f.write("invalid: yaml: syntax: here:\n  - bad indentation")
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError) as exc_info:
                RefactronConfig.from_file(config_path)

            error = exc_info.value
            assert "yaml" in str(error).lower()
            assert error.recovery_suggestion is not None
            assert "yaml" in error.recovery_suggestion.lower()
        finally:
            config_path.unlink()

    def test_valid_config_loads_successfully(self) -> None:
        """Test that valid config loads without error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("max_function_complexity: 15\n")
            f.write("enabled_analyzers:\n")
            f.write("  - complexity\n")
            config_path = Path(f.name)

        try:
            config = RefactronConfig.from_file(config_path)
            assert config.max_function_complexity == 15
            assert "complexity" in config.enabled_analyzers
        finally:
            config_path.unlink()


class TestRefactoringErrorHandling:
    """Test refactoring error handling with graceful degradation."""

    def test_refactor_continues_after_file_error(self) -> None:
        """Test that refactoring continues when one file fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create valid files
            valid_file1 = tmpdir_path / "valid1.py"
            valid_file1.write_text("def func():\n    x = 42\n    return x\n")

            # Create a file with invalid encoding
            invalid_file = tmpdir_path / "invalid.py"
            invalid_file.write_bytes(b"\xff\xfe\x00invalid")

            valid_file2 = tmpdir_path / "valid2.py"
            valid_file2.write_text("def func2():\n    y = 100\n    return y\n")

            refactron = Refactron()
            # Should not raise error, just log and continue
            result = refactron.refactor(tmpdir_path, preview=True)

            # Should have attempted all files, but may have different numbers
            # of operations based on which files succeeded
            assert result is not None


class TestAnalysisResultExtensions:
    """Test AnalysisResult enhancements."""

    def test_files_analyzed_successfully_property(self) -> None:
        """Test files_analyzed_successfully property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            file1 = tmpdir_path / "file1.py"
            file1.write_text("def func1():\n    pass\n")

            file2 = tmpdir_path / "file2.py"
            file2.write_text("def func2():\n    pass\n")

            refactron = Refactron()
            result = refactron.analyze(tmpdir_path)

            assert result.files_analyzed_successfully == 2
            assert result.files_failed == 0

    def test_files_failed_property(self) -> None:
        """Test files_failed property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            invalid_file = tmpdir_path / "invalid.py"
            invalid_file.write_bytes(b"\xff\xfe\x00invalid")

            refactron = Refactron()
            result = refactron.analyze(tmpdir_path)

            assert result.files_failed == 1
            assert result.files_analyzed_successfully == 0
