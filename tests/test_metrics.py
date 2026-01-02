"""Tests for metrics collection."""

import time

import pytest

from refactron.core.metrics import (
    FileMetric,
    MetricsCollector,
    RefactoringMetric,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestFileMetric:
    """Test FileMetric dataclass."""

    def test_creation(self):
        """Test creating a FileMetric."""
        metric = FileMetric(
            file_path="/path/to/file.py",
            analysis_time_ms=100.5,
            lines_of_code=50,
            issues_found=5,
            analyzers_run=["complexity", "security"],
            success=True,
        )

        assert metric.file_path == "/path/to/file.py"
        assert metric.analysis_time_ms == 100.5
        assert metric.lines_of_code == 50
        assert metric.issues_found == 5
        assert metric.analyzers_run == ["complexity", "security"]
        assert metric.success is True
        assert metric.error_message is None
        assert "timestamp" in metric.__dict__

    def test_failed_analysis(self):
        """Test FileMetric for failed analysis."""
        metric = FileMetric(
            file_path="/path/to/bad.py",
            analysis_time_ms=50.0,
            lines_of_code=0,
            issues_found=0,
            analyzers_run=[],
            success=False,
            error_message="Syntax error",
        )

        assert metric.success is False
        assert metric.error_message == "Syntax error"


class TestRefactoringMetric:
    """Test RefactoringMetric dataclass."""

    def test_creation(self):
        """Test creating a RefactoringMetric."""
        metric = RefactoringMetric(
            operation_type="extract_method",
            file_path="/path/to/file.py",
            execution_time_ms=200.0,
            success=True,
            risk_level="safe",
        )

        assert metric.operation_type == "extract_method"
        assert metric.file_path == "/path/to/file.py"
        assert metric.execution_time_ms == 200.0
        assert metric.success is True
        assert metric.risk_level == "safe"
        assert metric.error_message is None


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = MetricsCollector()

        assert collector.total_files_analyzed == 0
        assert collector.total_files_failed == 0
        assert collector.total_issues_found == 0
        assert len(collector.file_metrics) == 0
        assert len(collector.refactoring_metrics) == 0

    def test_record_file_analysis_success(self):
        """Test recording successful file analysis."""
        collector = MetricsCollector()

        collector.record_file_analysis(
            file_path="/path/to/file.py",
            analysis_time_ms=100.0,
            lines_of_code=50,
            issues_found=3,
            analyzers_run=["complexity"],
            success=True,
        )

        assert collector.total_files_analyzed == 1
        assert collector.total_files_failed == 0
        assert collector.total_issues_found == 3
        assert len(collector.file_metrics) == 1

    def test_record_file_analysis_failure(self):
        """Test recording failed file analysis."""
        collector = MetricsCollector()

        collector.record_file_analysis(
            file_path="/path/to/bad.py",
            analysis_time_ms=50.0,
            lines_of_code=0,
            issues_found=0,
            analyzers_run=[],
            success=False,
            error_message="Parse error",
        )

        assert collector.total_files_analyzed == 1
        assert collector.total_files_failed == 1
        assert collector.total_issues_found == 0

    def test_record_analyzer_hit(self):
        """Test recording analyzer hits."""
        collector = MetricsCollector()

        collector.record_analyzer_hit("complexity", "high_complexity")
        collector.record_analyzer_hit("complexity", "long_function")
        collector.record_analyzer_hit("security", "sql_injection")

        assert collector.analyzer_hits["complexity"] == 2
        assert collector.analyzer_hits["security"] == 1
        assert collector.issue_type_counts["high_complexity"] == 1
        assert collector.issue_type_counts["long_function"] == 1
        assert collector.issue_type_counts["sql_injection"] == 1

    def test_analysis_timing(self):
        """Test analysis start/end timing."""
        collector = MetricsCollector()

        collector.start_analysis()
        time.sleep(0.1)  # Simulate work
        collector.end_analysis()

        summary = collector.get_analysis_summary()
        assert summary["total_analysis_time_ms"] > 90.0  # At least 100ms

    def test_record_refactoring(self):
        """Test recording refactoring operations."""
        collector = MetricsCollector()

        collector.record_refactoring(
            operation_type="extract_method",
            file_path="/path/to/file.py",
            execution_time_ms=150.0,
            success=True,
            risk_level="safe",
        )

        assert collector.total_refactorings_applied == 1
        assert collector.total_refactorings_failed == 0
        assert collector.refactorer_hits["extract_method"] == 1

    def test_refactoring_timing(self):
        """Test refactoring start/end timing."""
        collector = MetricsCollector()

        collector.start_refactoring()
        time.sleep(0.1)  # Simulate work
        collector.end_refactoring()

        summary = collector.get_refactoring_summary()
        assert summary["total_refactoring_time_ms"] > 90.0

    def test_analysis_summary(self):
        """Test analysis summary generation."""
        collector = MetricsCollector()
        collector.start_analysis()

        # Record multiple file analyses
        collector.record_file_analysis(
            file_path="/file1.py",
            analysis_time_ms=100.0,
            lines_of_code=50,
            issues_found=2,
            analyzers_run=["complexity"],
            success=True,
        )
        collector.record_file_analysis(
            file_path="/file2.py",
            analysis_time_ms=200.0,
            lines_of_code=100,
            issues_found=5,
            analyzers_run=["security"],
            success=True,
        )

        collector.end_analysis()
        summary = collector.get_analysis_summary()

        assert summary["total_files_analyzed"] == 2
        assert summary["total_files_failed"] == 0
        assert summary["total_issues_found"] == 7
        assert summary["average_time_per_file_ms"] == 150.0
        assert summary["success_rate_percent"] == 100.0

    def test_refactoring_summary(self):
        """Test refactoring summary generation."""
        collector = MetricsCollector()
        collector.start_refactoring()

        # Record multiple refactorings
        collector.record_refactoring(
            operation_type="extract_method",
            file_path="/file1.py",
            execution_time_ms=100.0,
            success=True,
            risk_level="safe",
        )
        collector.record_refactoring(
            operation_type="simplify",
            file_path="/file2.py",
            execution_time_ms=150.0,
            success=True,
            risk_level="low",
        )
        collector.record_refactoring(
            operation_type="extract_method",
            file_path="/file3.py",
            execution_time_ms=80.0,
            success=False,
            risk_level="safe",
            error_message="Failed",
        )

        collector.end_refactoring()
        summary = collector.get_refactoring_summary()

        assert summary["total_refactorings_applied"] == 2
        assert summary["total_refactorings_failed"] == 1
        assert summary["average_time_per_operation_ms"] == pytest.approx(110.0, rel=0.1)
        assert summary["success_rate_percent"] == pytest.approx(66.67, rel=0.1)
        assert summary["risk_level_distribution"]["safe"] == 2
        assert summary["risk_level_distribution"]["low"] == 1

    def test_combined_summary(self):
        """Test combined summary generation."""
        collector = MetricsCollector()

        collector.start_analysis()
        collector.record_file_analysis(
            file_path="/file.py",
            analysis_time_ms=100.0,
            lines_of_code=50,
            issues_found=3,
            analyzers_run=["complexity"],
            success=True,
        )
        collector.end_analysis()

        collector.start_refactoring()
        collector.record_refactoring(
            operation_type="extract_method",
            file_path="/file.py",
            execution_time_ms=150.0,
            success=True,
        )
        collector.end_refactoring()

        summary = collector.get_combined_summary()

        assert "timestamp" in summary
        assert "analysis" in summary
        assert "refactoring" in summary
        assert summary["analysis"]["total_files_analyzed"] == 1
        assert summary["refactoring"]["total_refactorings_applied"] == 1

    def test_reset(self):
        """Test resetting metrics collector."""
        collector = MetricsCollector()

        # Add some metrics
        collector.start_analysis()
        collector.record_file_analysis(
            file_path="/file.py",
            analysis_time_ms=100.0,
            lines_of_code=50,
            issues_found=3,
            analyzers_run=["complexity"],
            success=True,
        )
        collector.end_analysis()

        # Reset
        collector.reset()

        # Verify everything is reset
        assert collector.total_files_analyzed == 0
        assert collector.total_files_failed == 0
        assert collector.total_issues_found == 0
        assert len(collector.file_metrics) == 0
        assert len(collector.analyzer_hits) == 0
        assert collector.analysis_start_time is None
        assert collector.analysis_end_time is None


def test_global_metrics_collector():
    """Test global metrics collector singleton."""
    # Reset to ensure clean state
    reset_metrics_collector()

    collector1 = get_metrics_collector()
    collector2 = get_metrics_collector()

    # Should return same instance
    assert collector1 is collector2

    # Test that metrics persist across calls
    collector1.record_file_analysis(
        file_path="/file.py",
        analysis_time_ms=100.0,
        lines_of_code=50,
        issues_found=3,
        analyzers_run=["complexity"],
        success=True,
    )

    assert collector2.total_files_analyzed == 1

    # Reset and verify
    reset_metrics_collector()
    collector3 = get_metrics_collector()
    assert collector3.total_files_analyzed == 0
