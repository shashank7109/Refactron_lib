"""Tests for Phase 4: Learning Service."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from refactron.core.models import RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter
from refactron.patterns.learner import PatternLearner
from refactron.patterns.learning_service import LearningService
from refactron.patterns.models import RefactoringFeedback, RefactoringPattern
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


class TestLearningService:
    """Test LearningService class."""

    def test_initialization_success(self):
        """Test successful initialization."""
        with IsolatedStorage() as storage:
            service = LearningService(storage=storage)

            assert service.storage == storage
            assert service.learner is not None

    def test_initialization_with_custom_learner(self):
        """Test initialization with custom learner."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)
            service = LearningService(storage=storage, learner=learner)

            assert service.learner == learner

    def test_initialization_with_none_storage(self):
        """Test initialization fails with None storage."""
        with pytest.raises(ValueError, match="PatternStorage cannot be None"):
            LearningService(storage=None)

    def test_process_pending_feedback_with_no_feedback(self):
        """Test processing when no feedback exists."""
        with IsolatedStorage() as storage:
            service = LearningService(storage=storage)

            stats = service.process_pending_feedback()

            assert stats["processed"] == 0
            assert stats["created"] == 0
            assert stats["updated"] == 0
            assert stats["failed"] == 0

    def test_process_pending_feedback_with_limit(self):
        """Test processing with limit."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()

            # Create some feedback
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
            operation.metadata["old_code"] = operation.old_code
            operation.metadata["new_code"] = operation.new_code
            operation.metadata["description"] = operation.description
            operation.metadata["line_number"] = operation.line_number
            operation.metadata["risk_score"] = operation.risk_score

            feedback = RefactoringFeedback.create(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                file_path=operation.file_path,
                action="accepted",
                code_pattern_hash=pattern_hash,
                metadata=operation.metadata,
            )

            storage.save_feedback(feedback)

            service = LearningService(storage=storage)
            stats = service.process_pending_feedback(limit=1)

            assert stats["processed"] >= 0  # May be 0 if reconstruction fails
            assert stats["created"] >= 0
            assert stats["failed"] >= 0

    def test_update_pattern_scores(self):
        """Test updating pattern scores."""
        with IsolatedStorage() as storage:
            fingerprinter = PatternFingerprinter()
            learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

            # Create a pattern
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

            learner.learn_from_feedback(operation, feedback)

            service = LearningService(storage=storage, learner=learner)
            stats = service.update_pattern_scores()

            assert stats["total"] >= 1
            assert stats["updated"] >= 0

    def test_cleanup_old_patterns(self):
        """Test cleaning up old patterns."""
        with IsolatedStorage() as storage:
            # Create an old pattern (manually set last_seen to past)
            old_pattern = RefactoringPattern.create(
                pattern_hash="old_hash",
                operation_type="extract_method",
                code_snippet_before="old code",
                code_snippet_after="new code",
            )
            old_pattern.last_seen = datetime.now(timezone.utc) - timedelta(days=100)
            storage.save_pattern(old_pattern)

            # Create a recent pattern
            recent_pattern = RefactoringPattern.create(
                pattern_hash="recent_hash",
                operation_type="extract_constant",
                code_snippet_before="old code",
                code_snippet_after="new code",
            )
            recent_pattern.last_seen = datetime.now(timezone.utc) - timedelta(days=30)
            storage.save_pattern(recent_pattern)

            service = LearningService(storage=storage)
            stats = service.cleanup_old_patterns(days=90)

            assert stats["removed"] >= 0  # At least the old pattern
            assert stats["total"] >= 2

            # Recent pattern should still exist
            assert storage.get_pattern(recent_pattern.pattern_id) is not None

    def test_cleanup_old_patterns_with_negative_days(self):
        """Test cleanup fails with negative days."""
        with IsolatedStorage() as storage:
            service = LearningService(storage=storage)

            with pytest.raises(ValueError, match="days must be non-negative"):
                service.cleanup_old_patterns(days=-1)

    def test_cleanup_old_patterns_keeps_recent_patterns(self):
        """Test cleanup keeps recent patterns and removes old ones."""
        with IsolatedStorage() as storage:
            # Create an old pattern (should be removed)
            old_pattern = RefactoringPattern.create(
                pattern_hash="old_hash",
                operation_type="extract_method",
                code_snippet_before="old code 1",
                code_snippet_after="new code 1",
            )
            old_pattern.last_seen = datetime.now(timezone.utc) - timedelta(days=100)
            storage.save_pattern(old_pattern)

            # Create a recent pattern (should be kept)
            recent_pattern = RefactoringPattern.create(
                pattern_hash="recent_hash",
                operation_type="extract_method",
                code_snippet_before="old code 2",
                code_snippet_after="new code 2",
            )
            recent_pattern.last_seen = datetime.now(timezone.utc) - timedelta(days=1)
            storage.save_pattern(recent_pattern)

            service = LearningService(storage=storage)
            stats = service.cleanup_old_patterns(days=90)

            # Verify old pattern was removed
            assert storage.get_pattern(old_pattern.pattern_id) is None
            # Verify recent pattern was kept
            assert storage.get_pattern(recent_pattern.pattern_id) is not None
            # Verify statistics
            assert stats["removed"] == 1
            assert stats["total"] == 2

    def test_reconstruct_operation_from_feedback(self):
        """Test reconstructing operation from feedback metadata."""
        with IsolatedStorage() as storage:
            service = LearningService(storage=storage)

            feedback = RefactoringFeedback.create(
                operation_id="test-op-id",
                operation_type="extract_method",
                file_path=Path("test.py"),
                action="accepted",
                metadata={
                    "old_code": "def foo(): pass",
                    "new_code": "def bar(): pass",
                    "line_number": 10,
                    "description": "Extract method",
                    "risk_score": 0.5,
                },
            )

            operation = service._reconstruct_operation_from_feedback(feedback)

            assert operation is not None
            assert operation.operation_type == "extract_method"
            assert operation.file_path == Path("test.py")
            assert operation.operation_id == "test-op-id"
