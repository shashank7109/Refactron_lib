"""Tests for Phase 4: Pattern Learning Engine."""

import tempfile
from pathlib import Path

import pytest

from refactron.core.models import FileMetrics, RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter
from refactron.patterns.learner import PatternLearner
from refactron.patterns.models import RefactoringFeedback
from refactron.patterns.storage import PatternStorage


class IsolatedStorage:
    """Context manager for isolated storage in tests."""

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = PatternStorage(storage_dir=Path(self.tmpdir))

    def __enter__(self):
        return self.storage

    def __exit__(self, *args):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestPatternLearner:
    """Test PatternLearner class."""

    def test_initialization_success(self):
        """Test successful initialization."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            assert learner.storage == storage
            assert learner.fingerprinter == fingerprinter

    def test_initialization_with_none_storage(self):
        """Test initialization fails with None storage."""
        fingerprinter = PatternFingerprinter()
        with pytest.raises(ValueError, match="PatternStorage cannot be None"):
            PatternLearner(storage=None, fingerprinter=fingerprinter)

    def test_initialization_with_none_fingerprinter(self):
        """Test initialization fails with None fingerprinter."""
        with IsolatedStorage() as storage:
            with pytest.raises(ValueError, match="PatternFingerprinter cannot be None"):
                PatternLearner(storage=storage, fingerprinter=None)

    def test_learn_from_feedback_creates_new_pattern(self):
        """Test learning from feedback creates a new pattern."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.5,
            )

            # Fingerprint the code
            pattern_hash = fingerprinter.fingerprint_code(operation.old_code)
            operation.metadata["code_pattern_hash"] = pattern_hash

            feedback = RefactoringFeedback.create(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                file_path=operation.file_path,
                action="accepted",
                code_pattern_hash=pattern_hash,
            )

            pattern_id = learner.learn_from_feedback(operation, feedback)

            assert pattern_id is not None
            pattern = storage.get_pattern(pattern_id)
            assert pattern is not None
            assert pattern.pattern_hash == pattern_hash
            assert pattern.operation_type == "extract_method"
            assert pattern.acceptance_rate > 0
            assert pattern.accepted_count == 1

    def test_learn_from_feedback_updates_existing_pattern(self):
        """Test learning from feedback updates existing pattern."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.5,
            )

            pattern_hash = fingerprinter.fingerprint_code(operation.old_code)
            operation.metadata["code_pattern_hash"] = pattern_hash

            # First feedback - accepted
            feedback1 = RefactoringFeedback.create(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                file_path=operation.file_path,
                action="accepted",
                code_pattern_hash=pattern_hash,
            )

            pattern_id1 = learner.learn_from_feedback(operation, feedback1)
            assert pattern_id1 is not None

            # Second feedback - rejected (same pattern, different operation)
            operation2 = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test2.py"),
                line_number=20,
                description="Extract method",
                old_code="def foo(): pass",  # Same code pattern
                new_code="def bar(): pass",
                risk_score=0.5,
            )
            operation2.metadata["code_pattern_hash"] = pattern_hash

            feedback2 = RefactoringFeedback.create(
                operation_id=operation2.operation_id,
                operation_type=operation2.operation_type,
                file_path=operation2.file_path,
                action="rejected",
                code_pattern_hash=pattern_hash,
            )

            pattern_id2 = learner.learn_from_feedback(operation2, feedback2)

            # Should be the same pattern
            assert pattern_id2 == pattern_id1

            # Check updated statistics (reload to get fresh data)
            pattern = storage.get_pattern(pattern_id1)
            assert pattern is not None
            assert pattern.accepted_count >= 1  # At least 1
            assert pattern.rejected_count >= 1  # At least 1
            assert pattern.total_occurrences >= 2  # At least 2
            # Acceptance rate should be calculated correctly
            total_decisions = pattern.accepted_count + pattern.rejected_count
            if total_decisions > 0:
                assert pattern.acceptance_rate == pattern.accepted_count / total_decisions

    def test_learn_from_feedback_with_none_operation(self):
        """Test learning fails with None operation."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            feedback = RefactoringFeedback.create(
                operation_id="test-id",
                operation_type="extract_method",
                file_path=Path("test.py"),
                action="accepted",
            )

            with pytest.raises(ValueError, match="RefactoringOperation cannot be None"):
                learner.learn_from_feedback(None, feedback)

    def test_learn_from_feedback_with_none_feedback(self):
        """Test learning fails with None feedback."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.5,
            )

            with pytest.raises(ValueError, match="RefactoringFeedback cannot be None"):
                learner.learn_from_feedback(operation, None)

    def test_batch_learn_processes_multiple_feedbacks(self):
        """Test batch learning processes multiple feedback records."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            # Create multiple operations and feedback
            operations_with_feedback = []

            for i in range(3):
                operation = RefactoringOperation(
                    operation_type="extract_method",
                    file_path=Path(f"test{i}.py"),
                    line_number=10 + i,
                    description="Extract method",
                    old_code=f"def foo{i}(): pass",
                    new_code=f"def bar{i}(): pass",
                    risk_score=0.5,
                )

                pattern_hash = fingerprinter.fingerprint_code(operation.old_code)
                operation.metadata["code_pattern_hash"] = pattern_hash

                feedback = RefactoringFeedback.create(
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type,
                    file_path=operation.file_path,
                    action="accepted" if i % 2 == 0 else "rejected",
                    code_pattern_hash=pattern_hash,
                )

                operations_with_feedback.append((operation, feedback))

            stats = learner.batch_learn(operations_with_feedback)

            assert stats["processed"] == 3
            # Anonymization makes structurally identical code produce the same hash
            # so all 3 will update the same pattern (1 created, 2 updated)
            assert stats["created"] == 1
            assert stats["updated"] == 2
            assert stats["failed"] == 0

    def test_batch_learn_with_none_list(self):
        """Test batch learning fails with None list."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            with pytest.raises(ValueError, match="feedback_list cannot be None"):
                learner.batch_learn(None)

    def test_update_pattern_metrics(self):
        """Test updating pattern metrics from before/after comparison."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            # Create a pattern first
            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.5,
            )

            pattern_hash = fingerprinter.fingerprint_code(operation.old_code)
            operation.metadata["code_pattern_hash"] = pattern_hash

            feedback = RefactoringFeedback.create(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                file_path=operation.file_path,
                action="accepted",
                code_pattern_hash=pattern_hash,
            )

            pattern_id = learner.learn_from_feedback(operation, feedback)
            assert pattern_id is not None

            # Create before/after metrics
            before_metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=100,
                comment_lines=10,
                blank_lines=5,
                complexity=15.0,
                maintainability_index=60.0,
                functions=10,
                classes=2,
            )
            before_metrics.issues = []  # Mock issues

            after_metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=95,
                comment_lines=12,
                blank_lines=5,
                complexity=12.0,  # Reduced
                maintainability_index=70.0,  # Improved
                functions=12,
                classes=2,
            )
            after_metrics.issues = []  # Mock issues

            # Update metrics
            learner.update_pattern_metrics(pattern_id, before_metrics, after_metrics)

            # Check metric was created
            metric = storage.get_pattern_metric(pattern_id)
            assert metric is not None
            assert metric.complexity_reduction == 3.0
            assert metric.maintainability_improvement == 10.0

    def test_update_pattern_metrics_with_invalid_pattern_id(self):
        """Test updating metrics with invalid pattern ID."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            before_metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=100,
                comment_lines=10,
                blank_lines=5,
                complexity=15.0,
                maintainability_index=60.0,
                functions=10,
                classes=2,
            )

            after_metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=95,
                comment_lines=12,
                blank_lines=5,
                complexity=12.0,
                maintainability_index=70.0,
                functions=12,
                classes=2,
            )

            # Should not raise, just log warning
            learner.update_pattern_metrics("non-existent-id", before_metrics, after_metrics)

    def test_update_pattern_metrics_with_empty_pattern_id(self):
        """Test updating metrics fails with empty pattern ID."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            before_metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=100,
                comment_lines=10,
                blank_lines=5,
                complexity=15.0,
                maintainability_index=60.0,
                functions=10,
                classes=2,
            )

            after_metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=95,
                comment_lines=12,
                blank_lines=5,
                complexity=12.0,
                maintainability_index=70.0,
                functions=12,
                classes=2,
            )

            with pytest.raises(ValueError, match="pattern_id cannot be empty"):
                learner.update_pattern_metrics("", before_metrics, after_metrics)

    def test_update_pattern_metrics_with_none_metrics(self):
        """Test updating metrics fails with None metrics."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            metrics = FileMetrics(
                file_path=Path("test.py"),
                lines_of_code=100,
                comment_lines=10,
                blank_lines=5,
                complexity=15.0,
                maintainability_index=60.0,
                functions=10,
                classes=2,
            )

            with pytest.raises(ValueError, match="before_metrics cannot be None"):
                learner.update_pattern_metrics("pattern-id", None, metrics)

            with pytest.raises(ValueError, match="after_metrics cannot be None"):
                learner.update_pattern_metrics("pattern-id", metrics, None)
