"""Custom exception types for Refactron.

This module defines granular exception types for different failure scenarios,
enabling better error handling and recovery strategies.
"""

from pathlib import Path
from typing import Optional


class RefactronError(Exception):
    """Base exception for all Refactron errors.

    All custom exceptions in Refactron inherit from this class,
    allowing for easy catching of all Refactron-specific errors.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            file_path: Optional path to the file that caused the error
            recovery_suggestion: Optional suggestion for how to recover from the error
        """
        self.file_path = file_path
        self.recovery_suggestion = recovery_suggestion
        super().__init__(message)

    def __str__(self) -> str:
        """Return a formatted error message."""
        msg = super().__str__()
        if self.file_path:
            msg = f"{msg} (file: {self.file_path})"
        if self.recovery_suggestion:
            msg = f"{msg}\n💡 Suggestion: {self.recovery_suggestion}"
        return msg


class AnalysisError(RefactronError):
    """Raised when code analysis fails.

    This exception is raised when an analyzer encounters an error
    while processing source code. Common causes include:
    - Syntax errors in the source code
    - Unsupported Python language features
    - AST parsing failures
    - File encoding issues
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        analyzer_name: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            file_path: Optional path to the file being analyzed
            analyzer_name: Name of the analyzer that failed
            recovery_suggestion: Optional suggestion for how to recover
        """
        self.analyzer_name = analyzer_name

        # Provide default recovery suggestion if not specified
        if not recovery_suggestion:
            if "syntax" in message.lower():
                recovery_suggestion = "Check the file for syntax errors using a Python linter"
            elif "encoding" in message.lower():
                recovery_suggestion = "Ensure the file is saved with UTF-8 encoding"
            else:
                recovery_suggestion = "Verify the file contains valid Python code"

        super().__init__(message, file_path, recovery_suggestion)

    def __str__(self) -> str:
        """Return a formatted error message with optional analyzer context."""
        base_message = Exception.__str__(self)
        if self.analyzer_name:
            base_message = f"[{self.analyzer_name}] {base_message}"

        # Use parent formatting by temporarily updating the message
        msg = base_message
        if self.file_path:
            msg = f"{msg} (file: {self.file_path})"
        if self.recovery_suggestion:
            msg = f"{msg}\n💡 Suggestion: {self.recovery_suggestion}"
        return msg


class RefactoringError(RefactronError):
    """Raised when code refactoring fails.

    This exception is raised when a refactoring operation cannot be
    completed successfully. Common causes include:
    - Unable to parse the source code
    - Refactoring would break code semantics
    - File write permission issues
    - Backup creation failures
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[Path] = None,
        operation_type: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            file_path: Optional path to the file being refactored
            operation_type: Type of refactoring operation that failed
            recovery_suggestion: Optional suggestion for how to recover
        """
        self.operation_type = operation_type

        # Provide default recovery suggestion if not specified
        if not recovery_suggestion:
            if "permission" in message.lower():
                recovery_suggestion = "Check file permissions and ensure you have write access"
            elif "backup" in message.lower():
                recovery_suggestion = (
                    "Ensure sufficient disk space and write permissions " "for backup directory"
                )
            else:
                recovery_suggestion = (
                    "Try running the operation on a single file first " "to identify the issue"
                )

        super().__init__(message, file_path, recovery_suggestion)

    def __str__(self) -> str:
        """Return a formatted error message with optional operation type context."""
        base_message = Exception.__str__(self)
        if self.operation_type:
            base_message = f"[{self.operation_type}] {base_message}"

        # Use parent formatting by temporarily updating the message
        msg = base_message
        if self.file_path:
            msg = f"{msg} (file: {self.file_path})"
        if self.recovery_suggestion:
            msg = f"{msg}\n💡 Suggestion: {self.recovery_suggestion}"
        return msg


class ConfigError(RefactronError):
    """Raised when configuration is invalid or cannot be loaded.

    This exception is raised when there are problems with the
    configuration. Common causes include:
    - Invalid YAML syntax in config file
    - Missing required configuration options
    - Invalid configuration values (e.g., negative thresholds)
    - Configuration file not found or not readable
    """

    def __init__(
        self,
        message: str,
        config_path: Optional[Path] = None,
        config_key: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            config_path: Optional path to the config file
            config_key: Optional specific config key that caused the error
            recovery_suggestion: Optional suggestion for how to recover
        """
        self.config_key = config_key

        # Provide default recovery suggestion if not specified
        if not recovery_suggestion:
            if "not found" in message.lower() or "does not exist" in message.lower():
                recovery_suggestion = (
                    "Create a config file using 'refactron init' " "or use default configuration"
                )
            elif "yaml" in message.lower() or "syntax" in message.lower():
                recovery_suggestion = "Check the YAML syntax in your configuration file"
            elif config_key:
                recovery_suggestion = f"Check the value for '{config_key}' in your configuration"
            else:
                recovery_suggestion = "Verify your configuration file follows the expected format"

        super().__init__(message, config_path, recovery_suggestion)

    def __str__(self) -> str:
        """Return a formatted error message with optional config key context."""
        base_message = Exception.__str__(self)
        if self.config_key:
            base_message = f"[config key: {self.config_key}] {base_message}"

        # Use parent formatting by temporarily updating the message
        msg = base_message
        if self.file_path:
            msg = f"{msg} (file: {self.file_path})"
        if self.recovery_suggestion:
            msg = f"{msg}\n💡 Suggestion: {self.recovery_suggestion}"
        return msg
