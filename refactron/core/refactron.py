"""Main Refactron class - the entry point for all operations."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from refactron.analyzers.base_analyzer import BaseAnalyzer
from refactron.analyzers.code_smell_analyzer import CodeSmellAnalyzer
from refactron.analyzers.complexity_analyzer import ComplexityAnalyzer
from refactron.analyzers.dead_code_analyzer import DeadCodeAnalyzer
from refactron.analyzers.dependency_analyzer import DependencyAnalyzer
from refactron.analyzers.performance_analyzer import PerformanceAnalyzer
from refactron.analyzers.security_analyzer import SecurityAnalyzer
from refactron.analyzers.type_hint_analyzer import TypeHintAnalyzer
from refactron.core.analysis_result import AnalysisResult, FileAnalysisError
from refactron.core.config import RefactronConfig
from refactron.core.exceptions import AnalysisError, RefactoringError
from refactron.core.models import FileMetrics
from refactron.core.refactor_result import RefactorResult
from refactron.refactorers.add_docstring_refactorer import AddDocstringRefactorer
from refactron.refactorers.base_refactorer import BaseRefactorer
from refactron.refactorers.extract_method_refactorer import ExtractMethodRefactorer
from refactron.refactorers.magic_number_refactorer import MagicNumberRefactorer
from refactron.refactorers.reduce_parameters_refactorer import ReduceParametersRefactorer
from refactron.refactorers.simplify_conditionals_refactorer import SimplifyConditionalsRefactorer

# Configure logging
logger = logging.getLogger(__name__)


class Refactron:
    """
    Main Refactron class for code analysis and refactoring.

    Example:
        >>> refactron = Refactron()
        >>> result = refactron.analyze("mycode.py")
        >>> print(result.report())
    """

    def __init__(self, config: Optional[RefactronConfig] = None):
        """
        Initialize Refactron.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or RefactronConfig.default()
        self.analyzers: List[BaseAnalyzer] = []
        self.refactorers: List[BaseRefactorer] = []
        self._initialize_analyzers()
        self._initialize_refactorers()

    def _initialize_analyzers(self) -> None:
        """Initialize all enabled analyzers."""
        if "complexity" in self.config.enabled_analyzers:
            self.analyzers.append(ComplexityAnalyzer(self.config))

        if "code_smells" in self.config.enabled_analyzers:
            self.analyzers.append(CodeSmellAnalyzer(self.config))

        if "security" in self.config.enabled_analyzers:
            self.analyzers.append(SecurityAnalyzer(self.config))

        if "dependency" in self.config.enabled_analyzers:
            self.analyzers.append(DependencyAnalyzer(self.config))

        if "dead_code" in self.config.enabled_analyzers:
            self.analyzers.append(DeadCodeAnalyzer(self.config))

        if "type_hints" in self.config.enabled_analyzers:
            self.analyzers.append(TypeHintAnalyzer(self.config))

        if "performance" in self.config.enabled_analyzers:
            self.analyzers.append(PerformanceAnalyzer(self.config))

    def _initialize_refactorers(self) -> None:
        """Initialize all enabled refactorers."""
        if "extract_method" in self.config.enabled_refactorers:
            self.refactorers.append(ExtractMethodRefactorer(self.config))

        if "extract_constant" in self.config.enabled_refactorers:
            self.refactorers.append(MagicNumberRefactorer(self.config))

        if "simplify_conditionals" in self.config.enabled_refactorers:
            self.refactorers.append(SimplifyConditionalsRefactorer(self.config))

        if "reduce_parameters" in self.config.enabled_refactorers:
            self.refactorers.append(ReduceParametersRefactorer(self.config))

        if "add_docstring" in self.config.enabled_refactorers:
            self.refactorers.append(AddDocstringRefactorer(self.config))

    def analyze(self, target: Union[str, Path]) -> AnalysisResult:
        """
        Analyze a file or directory.

        Args:
            target: Path to file or directory to analyze

        Returns:
            AnalysisResult containing all detected issues and any errors encountered

        Note:
            This method implements graceful degradation - if individual files fail
            to analyze, they are logged and skipped, allowing analysis to continue
            on remaining files.
        """
        target_path = Path(target)

        if not target_path.exists():
            raise FileNotFoundError(f"Target not found: {target}")

        if target_path.is_file():
            files = [target_path]
        else:
            files = self._get_python_files(target_path)

        result = AnalysisResult(total_files=len(files))

        for file_path in files:
            try:
                file_metrics = self._analyze_file(file_path)
                result.file_metrics.append(file_metrics)
                result.total_issues += file_metrics.issue_count
            except AnalysisError as e:
                # Log the error and add to failed files list
                logger.debug(f"Failed to analyze {file_path}: {e}")
                result.failed_files.append(
                    FileAnalysisError(
                        file_path=file_path,
                        error_message=str(e),
                        error_type=e.__class__.__name__,
                        recovery_suggestion=e.recovery_suggestion,
                    )
                )
            except Exception as e:
                # Catch unexpected errors to ensure analysis continues
                logger.error(f"Unexpected error analyzing {file_path}: {e}", exc_info=True)
                result.failed_files.append(
                    FileAnalysisError(
                        file_path=file_path,
                        error_message=str(e),
                        error_type=e.__class__.__name__,
                        recovery_suggestion="Check the file for syntax errors or encoding issues",
                    )
                )

        return result

    def _analyze_file(self, file_path: Path) -> FileMetrics:
        """Analyze a single file.

        Args:
            file_path: Path to file to analyze

        Returns:
            FileMetrics with analysis results

        Raises:
            AnalysisError: If file cannot be analyzed
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except UnicodeDecodeError as e:
            raise AnalysisError(
                f"Failed to read file due to encoding error: {e}",
                file_path=file_path,
            ) from e
        except (IOError, OSError) as e:
            raise AnalysisError(
                f"Failed to read file: {e}",
                file_path=file_path,
            ) from e

        # Initialize basic metrics
        try:
            lines = source_code.split("\n")
            loc = len([line for line in lines if line.strip() and not line.strip().startswith("#")])
            comment_lines = len([line for line in lines if line.strip().startswith("#")])
            blank_lines = len([line for line in lines if not line.strip()])

            metrics = FileMetrics(
                file_path=file_path,
                lines_of_code=loc,
                comment_lines=comment_lines,
                blank_lines=blank_lines,
                complexity=0.0,
                maintainability_index=100.0,
                functions=0,
                classes=0,
            )
        except Exception as e:
            raise AnalysisError(
                f"Failed to compute basic metrics: {e}",
                file_path=file_path,
            ) from e

        # Run all analyzers with individual error handling
        for analyzer in self.analyzers:
            try:
                issues = analyzer.analyze(file_path, source_code)
                metrics.issues.extend(issues)
            except Exception as e:
                # Log analyzer failure but continue with other analyzers
                # Use debug level for expected errors, warning for unexpected
                if isinstance(e, AnalysisError):
                    logger.debug(f"Analyzer {analyzer.name} failed for {file_path}: {e}")
                else:
                    logger.warning(
                        f"Analyzer {analyzer.name} failed for {file_path}: {e}",
                        exc_info=True,
                    )
                # Don't raise - allow other analyzers to run

        return metrics

    def refactor(
        self,
        target: Union[str, Path],
        preview: bool = True,
        operation_types: Optional[List[str]] = None,
    ) -> RefactorResult:
        """
        Refactor a file or directory.

        Args:
            target: Path to file or directory to refactor
            preview: If True, show changes without applying them
            operation_types: Specific refactoring operations to apply (None = all)

        Returns:
            RefactorResult containing all proposed operations

        Note:
            This method implements graceful degradation - if individual files fail
            to refactor, they are logged and skipped, allowing refactoring to continue
            on remaining files.
        """
        target_path = Path(target)

        if not target_path.exists():
            raise FileNotFoundError(f"Target not found: {target}")

        if target_path.is_file():
            files = [target_path]
        else:
            files = self._get_python_files(target_path)

        result = RefactorResult(preview_mode=preview)

        for file_path in files:
            try:
                operations = self._refactor_file(file_path, operation_types)
                result.operations.extend(operations)
            except RefactoringError as e:
                # Log the error and continue with other files
                logger.warning(f"Failed to refactor {file_path}: {e}")
                # Store error in result if needed (RefactorResult could be extended)
            except Exception as e:
                # Catch unexpected errors to ensure refactoring continues
                logger.error(f"Unexpected error refactoring {file_path}: {e}", exc_info=True)

        return result

    def _refactor_file(
        self,
        file_path: Path,
        operation_types: Optional[List[str]] = None,
    ) -> List:
        """Refactor a single file.

        Args:
            file_path: Path to file to refactor
            operation_types: Specific operation types to apply

        Returns:
            List of refactoring operations

        Raises:
            RefactoringError: If file cannot be refactored
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except UnicodeDecodeError as e:
            raise RefactoringError(
                f"Failed to read file due to encoding error: {e}",
                file_path=file_path,
            ) from e
        except (IOError, OSError) as e:
            raise RefactoringError(
                f"Failed to read file: {e}",
                file_path=file_path,
            ) from e

        operations = []

        # Run all refactorers with individual error handling
        for refactorer in self.refactorers:
            if operation_types and refactorer.operation_type not in operation_types:
                continue

            try:
                ops = refactorer.refactor(file_path, source_code)
                operations.extend(ops)
            except Exception as e:
                # Log refactorer failure but continue with other refactorers
                logger.warning(
                    f"Refactorer {refactorer.__class__.__name__} failed for {file_path}: {e}",
                    exc_info=True,
                )
                # Don't raise - allow other refactorers to run

        return operations

    def _get_python_files(self, directory: Path) -> List[Path]:
        """Get all Python files in a directory, respecting exclude patterns."""
        python_files = []

        for root, dirs, files in os.walk(directory):
            root_path = Path(root)

            # Check if this directory should be excluded
            if self._should_exclude(root_path):
                dirs.clear()  # Don't descend into this directory
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = root_path / file
                    if not self._should_exclude(file_path):
                        python_files.append(file_path)

        return python_files

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded based on patterns."""
        path_str = str(path)

        for pattern in self.config.exclude_patterns:
            # Simple pattern matching
            pattern_clean = pattern.replace("**/", "").replace("/**", "")
            if pattern_clean in path_str:
                return True

        return False
