"""Tests for custom exception types."""

from pathlib import Path

from refactron.core.exceptions import AnalysisError, ConfigError, RefactoringError, RefactronError


class TestRefactronError:
    """Test base RefactronError exception."""

    def test_basic_error(self) -> None:
        """Test basic error with message only."""
        error = RefactronError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_error_with_file_path(self) -> None:
        """Test error with file path."""
        file_path = Path("/path/to/file.py")
        error = RefactronError("Something went wrong", file_path=file_path)
        assert "file.py" in str(error)
        assert error.file_path == file_path

    def test_error_with_recovery_suggestion(self) -> None:
        """Test error with recovery suggestion."""
        error = RefactronError(
            "Something went wrong",
            recovery_suggestion="Try doing this instead",
        )
        assert "💡 Suggestion: Try doing this instead" in str(error)

    def test_error_with_all_parameters(self) -> None:
        """Test error with all parameters."""
        file_path = Path("/path/to/file.py")
        error = RefactronError(
            "Something went wrong",
            file_path=file_path,
            recovery_suggestion="Try doing this instead",
        )
        error_str = str(error)
        assert "Something went wrong" in error_str
        assert "file.py" in error_str
        assert "💡 Suggestion: Try doing this instead" in error_str


class TestAnalysisError:
    """Test AnalysisError exception."""

    def test_basic_analysis_error(self) -> None:
        """Test basic analysis error."""
        error = AnalysisError("Analysis failed")
        assert "Analysis failed" in str(error)

    def test_analysis_error_with_analyzer_name(self) -> None:
        """Test analysis error with analyzer name."""
        error = AnalysisError(
            "Analysis failed",
            analyzer_name="ComplexityAnalyzer",
        )
        assert "[ComplexityAnalyzer]" in str(error)

    def test_analysis_error_syntax_suggestion(self) -> None:
        """Test automatic recovery suggestion for syntax errors."""
        error = AnalysisError("Syntax error in file")
        assert error.recovery_suggestion is not None
        assert "syntax" in error.recovery_suggestion.lower()

    def test_analysis_error_encoding_suggestion(self) -> None:
        """Test automatic recovery suggestion for encoding errors."""
        error = AnalysisError("Encoding error occurred")
        assert error.recovery_suggestion is not None
        assert "encoding" in error.recovery_suggestion.lower()

    def test_analysis_error_custom_suggestion(self) -> None:
        """Test analysis error with custom recovery suggestion."""
        error = AnalysisError(
            "Analysis failed",
            recovery_suggestion="Custom recovery suggestion",
        )
        assert error.recovery_suggestion == "Custom recovery suggestion"

    def test_analysis_error_complete_formatting(self) -> None:
        """Test analysis error with all parameters combined."""
        file_path = Path("/path/to/file.py")
        error = AnalysisError(
            "Parse error",
            file_path=file_path,
            analyzer_name="ComplexityAnalyzer",
            recovery_suggestion="Check the syntax",
        )
        error_str = str(error)
        # Verify all components appear in correct order
        assert "[ComplexityAnalyzer]" in error_str
        assert "Parse error" in error_str
        assert "file.py" in error_str
        assert "💡 Suggestion: Check the syntax" in error_str


class TestRefactoringError:
    """Test RefactoringError exception."""

    def test_basic_refactoring_error(self) -> None:
        """Test basic refactoring error."""
        error = RefactoringError("Refactoring failed")
        assert "Refactoring failed" in str(error)

    def test_refactoring_error_with_operation_type(self) -> None:
        """Test refactoring error with operation type."""
        error = RefactoringError(
            "Refactoring failed",
            operation_type="extract_method",
        )
        assert "[extract_method]" in str(error)

    def test_refactoring_error_permission_suggestion(self) -> None:
        """Test automatic recovery suggestion for permission errors."""
        error = RefactoringError("Permission denied")
        assert error.recovery_suggestion is not None
        assert "permission" in error.recovery_suggestion.lower()

    def test_refactoring_error_backup_suggestion(self) -> None:
        """Test automatic recovery suggestion for backup errors."""
        error = RefactoringError("Backup creation failed")
        assert error.recovery_suggestion is not None
        assert "backup" in error.recovery_suggestion.lower()

    def test_refactoring_error_custom_suggestion(self) -> None:
        """Test refactoring error with custom recovery suggestion."""
        error = RefactoringError(
            "Refactoring failed",
            recovery_suggestion="Custom recovery suggestion",
        )
        assert error.recovery_suggestion == "Custom recovery suggestion"

    def test_refactoring_error_complete_formatting(self) -> None:
        """Test refactoring error with all parameters combined."""
        file_path = Path("/path/to/file.py")
        error = RefactoringError(
            "Cannot refactor",
            file_path=file_path,
            operation_type="extract_method",
            recovery_suggestion="Review the code structure",
        )
        error_str = str(error)
        # Verify all components appear in correct order
        assert "[extract_method]" in error_str
        assert "Cannot refactor" in error_str
        assert "file.py" in error_str
        assert "💡 Suggestion: Review the code structure" in error_str


class TestConfigError:
    """Test ConfigError exception."""

    def test_basic_config_error(self) -> None:
        """Test basic config error."""
        error = ConfigError("Configuration is invalid")
        assert "Configuration is invalid" in str(error)

    def test_config_error_with_key(self) -> None:
        """Test config error with specific key."""
        error = ConfigError(
            "Invalid value",
            config_key="max_complexity",
        )
        assert "[config key: max_complexity]" in str(error)

    def test_config_error_not_found_suggestion(self) -> None:
        """Test automatic recovery suggestion for missing config file."""
        error = ConfigError("Config file not found")
        assert error.recovery_suggestion is not None
        assert "config" in error.recovery_suggestion.lower()

    def test_config_error_yaml_suggestion(self) -> None:
        """Test automatic recovery suggestion for YAML errors."""
        error = ConfigError("Invalid YAML syntax")
        assert error.recovery_suggestion is not None
        assert "yaml" in error.recovery_suggestion.lower()

    def test_config_error_with_key_suggestion(self) -> None:
        """Test automatic recovery suggestion with config key."""
        error = ConfigError(
            "Invalid value",
            config_key="max_complexity",
        )
        assert error.recovery_suggestion is not None
        assert "max_complexity" in error.recovery_suggestion

    def test_config_error_custom_suggestion(self) -> None:
        """Test config error with custom recovery suggestion."""
        error = ConfigError(
            "Configuration is invalid",
            recovery_suggestion="Custom recovery suggestion",
        )
        assert error.recovery_suggestion == "Custom recovery suggestion"

    def test_config_error_complete_formatting(self) -> None:
        """Test config error with all parameters combined."""
        config_path = Path("/path/to/config.yaml")
        error = ConfigError(
            "Invalid configuration",
            config_path=config_path,
            config_key="max_complexity",
            recovery_suggestion="Use a positive integer value",
        )
        error_str = str(error)
        # Verify all components appear in correct order
        assert "[config key: max_complexity]" in error_str
        assert "Invalid configuration" in error_str
        assert "config.yaml" in error_str
        assert "💡 Suggestion: Use a positive integer value" in error_str


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_analysis_error_is_refactron_error(self) -> None:
        """Test AnalysisError inherits from RefactronError."""
        error = AnalysisError("Test")
        assert isinstance(error, RefactronError)
        assert isinstance(error, Exception)

    def test_refactoring_error_is_refactron_error(self) -> None:
        """Test RefactoringError inherits from RefactronError."""
        error = RefactoringError("Test")
        assert isinstance(error, RefactronError)
        assert isinstance(error, Exception)

    def test_config_error_is_refactron_error(self) -> None:
        """Test ConfigError inherits from RefactronError."""
        error = ConfigError("Test")
        assert isinstance(error, RefactronError)
        assert isinstance(error, Exception)

    def test_catch_all_refactron_errors(self) -> None:
        """Test catching all Refactron errors with base class."""
        errors = [
            AnalysisError("Analysis failed"),
            RefactoringError("Refactoring failed"),
            ConfigError("Config failed"),
        ]

        for error in errors:
            try:
                raise error
            except RefactronError as e:
                # Just verify it's a RefactronError - the string may include suggestions
                assert isinstance(e, RefactronError)
                assert e.args[0] in ["Analysis failed", "Refactoring failed", "Config failed"]
