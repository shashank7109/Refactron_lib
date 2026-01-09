"""Metrics collection and tracking for Refactron.

This module provides execution metrics tracking including:
- Analysis time per file and total run time
- Refactoring success/failure rates
- Rule hit counts per analyzer/refactorer
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class FileMetric:
    """Metrics for a single file analysis."""

    file_path: str
    analysis_time_ms: float
    lines_of_code: int
    issues_found: int
    analyzers_run: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class RefactoringMetric:
    """Metrics for a single refactoring operation."""

    operation_type: str
    file_path: str
    execution_time_ms: float
    success: bool
    risk_level: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    error_message: Optional[str] = None


class MetricsCollector:
    """Centralized metrics collection for Refactron operations."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = Lock()
        
        # Analysis metrics
        self.file_metrics: List[FileMetric] = []
        self.analysis_start_time: Optional[float] = None
        self.analysis_end_time: Optional[float] = None
        
        # Refactoring metrics
        self.refactoring_metrics: List[RefactoringMetric] = []
        self.refactoring_start_time: Optional[float] = None
        self.refactoring_end_time: Optional[float] = None
        
        # Rule hit counts
        self.analyzer_hits: Counter = Counter()
        self.refactorer_hits: Counter = Counter()
        self.issue_type_counts: Counter = Counter()
        
        # Summary statistics
        self.total_files_analyzed: int = 0
        self.total_files_failed: int = 0
        self.total_issues_found: int = 0
        self.total_refactorings_applied: int = 0
        self.total_refactorings_failed: int = 0

    def start_analysis(self) -> None:
        """Mark the start of an analysis session."""
        with self._lock:
            self.analysis_start_time = time.time()

    def end_analysis(self) -> None:
        """Mark the end of an analysis session."""
        with self._lock:
            self.analysis_end_time = time.time()

    def record_file_analysis(
        self,
        file_path: str,
        analysis_time_ms: float,
        lines_of_code: int,
        issues_found: int,
        analyzers_run: List[str],
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Record metrics for a single file analysis.

        Args:
            file_path: Path to the analyzed file
            analysis_time_ms: Time taken to analyze the file in milliseconds
            lines_of_code: Number of lines of code in the file
            issues_found: Number of issues found in the file
            analyzers_run: List of analyzer names that were run
            success: Whether the analysis succeeded
            error_message: Error message if analysis failed
        """
        with self._lock:
            metric = FileMetric(
                file_path=file_path,
                analysis_time_ms=analysis_time_ms,
                lines_of_code=lines_of_code,
                issues_found=issues_found,
                analyzers_run=analyzers_run,
                success=success,
                error_message=error_message,
            )
            self.file_metrics.append(metric)
            
            self.total_files_analyzed += 1
            if not success:
                self.total_files_failed += 1
            self.total_issues_found += issues_found

            # Update analyzer hit counts
            for analyzer in analyzers_run:
                self.analyzer_hits[analyzer] += 1

    def record_analyzer_hit(self, analyzer_name: str, issue_type: str) -> None:
        """Record that an analyzer found an issue.

        Args:
            analyzer_name: Name of the analyzer
            issue_type: Type of issue found
        """
        with self._lock:
            self.analyzer_hits[analyzer_name] += 1
            self.issue_type_counts[issue_type] += 1

    def start_refactoring(self) -> None:
        """Mark the start of a refactoring session."""
        with self._lock:
            self.refactoring_start_time = time.time()

    def end_refactoring(self) -> None:
        """Mark the end of a refactoring session."""
        with self._lock:
            self.refactoring_end_time = time.time()

    def record_refactoring(
        self,
        operation_type: str,
        file_path: str,
        execution_time_ms: float,
        success: bool,
        risk_level: str = "safe",
        error_message: Optional[str] = None,
    ) -> None:
        """Record metrics for a single refactoring operation.

        Args:
            operation_type: Type of refactoring operation
            file_path: Path to the refactored file
            execution_time_ms: Time taken to perform refactoring in milliseconds
            success: Whether the refactoring succeeded
            risk_level: Risk level of the refactoring
            error_message: Error message if refactoring failed
        """
        with self._lock:
            metric = RefactoringMetric(
                operation_type=operation_type,
                file_path=file_path,
                execution_time_ms=execution_time_ms,
                success=success,
                risk_level=risk_level,
                error_message=error_message,
            )
            self.refactoring_metrics.append(metric)
            
            if success:
                self.total_refactorings_applied += 1
                self.refactorer_hits[operation_type] += 1
            else:
                self.total_refactorings_failed += 1

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis metrics.

        Returns:
            Dictionary containing analysis summary metrics
        """
        with self._lock:
            total_time_ms = 0.0
            if self.analysis_start_time and self.analysis_end_time:
                total_time_ms = (self.analysis_end_time - self.analysis_start_time) * 1000

            avg_time_per_file_ms = 0.0
            if self.file_metrics:
                avg_time_per_file_ms = sum(m.analysis_time_ms for m in self.file_metrics) / len(
                    self.file_metrics
                )

            success_rate = 0.0
            if self.total_files_analyzed > 0:
                success_rate = (
                    (self.total_files_analyzed - self.total_files_failed)
                    / self.total_files_analyzed
                ) * 100

            return {
                "total_files_analyzed": self.total_files_analyzed,
                "total_files_failed": self.total_files_failed,
                "total_issues_found": self.total_issues_found,
                "total_analysis_time_ms": total_time_ms,
                "average_time_per_file_ms": avg_time_per_file_ms,
                "success_rate_percent": success_rate,
                "analyzer_hit_counts": dict(self.analyzer_hits),
                "issue_type_counts": dict(self.issue_type_counts),
            }

    def get_refactoring_summary(self) -> Dict[str, Any]:
        """Get summary of refactoring metrics.

        Returns:
            Dictionary containing refactoring summary metrics
        """
        with self._lock:
            total_time_ms = 0.0
            if self.refactoring_start_time and self.refactoring_end_time:
                total_time_ms = (self.refactoring_end_time - self.refactoring_start_time) * 1000

            avg_time_per_operation_ms = 0.0
            if self.refactoring_metrics:
                avg_time_per_operation_ms = sum(
                    m.execution_time_ms for m in self.refactoring_metrics
                ) / len(self.refactoring_metrics)

            success_rate = 0.0
            total_operations = self.total_refactorings_applied + self.total_refactorings_failed
            if total_operations > 0:
                success_rate = (self.total_refactorings_applied / total_operations) * 100

            # Group by risk level
            risk_level_counts: Dict[str, int] = defaultdict(int)
            for metric in self.refactoring_metrics:
                risk_level_counts[metric.risk_level] += 1

            return {
                "total_refactorings_applied": self.total_refactorings_applied,
                "total_refactorings_failed": self.total_refactorings_failed,
                "total_refactoring_time_ms": total_time_ms,
                "average_time_per_operation_ms": avg_time_per_operation_ms,
                "success_rate_percent": success_rate,
                "refactorer_hit_counts": dict(self.refactorer_hits),
                "risk_level_distribution": dict(risk_level_counts),
            }

    def get_combined_summary(self) -> Dict[str, Any]:
        """Get combined summary of all metrics.

        Returns:
            Dictionary containing all metrics summaries
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "analysis": self.get_analysis_summary(),
            "refactoring": self.get_refactoring_summary(),
        }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self.file_metrics.clear()
            self.refactoring_metrics.clear()
            self.analyzer_hits.clear()
            self.refactorer_hits.clear()
            self.issue_type_counts.clear()
            
            self.analysis_start_time = None
            self.analysis_end_time = None
            self.refactoring_start_time = None
            self.refactoring_end_time = None
            
            self.total_files_analyzed = 0
            self.total_files_failed = 0
            self.total_issues_found = 0
            self.total_refactorings_applied = 0
            self.total_refactorings_failed = 0


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance
    """
    global _global_metrics_collector
    with _collector_lock:
        if _global_metrics_collector is None:
            _global_metrics_collector = MetricsCollector()
        return _global_metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _global_metrics_collector
    with _collector_lock:
        if _global_metrics_collector is not None:
            _global_metrics_collector.reset()
