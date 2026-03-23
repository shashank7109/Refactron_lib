"""
Refactron - The Intelligent Code Refactoring Transformer

A powerful Python library for code refactoring, optimization, and technical debt elimination.
"""

from typing import TYPE_CHECKING, Any

__version__ = "1.0.15"
__author__ = "Om Sherikar"

__all__ = [
    "Refactron",
    "AnalysisResult",
    "RefactorResult",
    "RefactronError",
    "AnalysisError",
    "RefactoringError",
    "ConfigError",
]

if TYPE_CHECKING:
    from refactron.core.analysis_result import AnalysisResult
    from refactron.core.exceptions import (
        AnalysisError,
        ConfigError,
        RefactoringError,
        RefactronError,
    )
    from refactron.core.refactor_result import RefactorResult
    from refactron.core.refactron import Refactron


def __getattr__(name: str) -> Any:
    """
    Lazily load heavy public symbols.
    This keeps lightweight CLI paths (e.g. `--version`) fast and side-effect free.
    """
    if name == "Refactron":
        from refactron.core.refactron import Refactron

        return Refactron
    if name == "AnalysisResult":
        from refactron.core.analysis_result import AnalysisResult

        return AnalysisResult
    if name == "RefactorResult":
        from refactron.core.refactor_result import RefactorResult

        return RefactorResult
    if name in {"RefactronError", "AnalysisError", "RefactoringError", "ConfigError"}:
        from refactron.core.exceptions import (
            AnalysisError,
            ConfigError,
            RefactoringError,
            RefactronError,
        )

        return {
            "RefactronError": RefactronError,
            "AnalysisError": AnalysisError,
            "RefactoringError": RefactoringError,
            "ConfigError": ConfigError,
        }[name]
    raise AttributeError(f"module 'refactron' has no attribute '{name}'")
