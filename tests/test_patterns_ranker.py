"""Tests for RefactoringRanker class."""

import tempfile
from pathlib import Path

import pytest

from refactron.core.models import RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter
from refactron.patterns.matcher import PatternMatcher
from refactron.patterns.models import RefactoringPattern
from refactron.patterns.ranker import RefactoringRanker
from refactron.patterns.storage import PatternStorage


class TestRefactoringRanker:
    """Test RefactoringRanker class."""

    def test_init_requires_all_dependencies(self):
        """Test that ranker requires all dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()

            # Should succeed with all dependencies
            ranker = RefactoringRanker(storage, matcher, fingerprinter)
            assert ranker.storage == storage
            assert ranker.matcher == matcher
            assert ranker.fingerprinter == fingerprinter

            # Should fail with None dependencies
            with pytest.raises(ValueError, match="PatternStorage cannot be None"):
                RefactoringRanker(None, matcher, fingerprinter)

            with pytest.raises(ValueError, match="PatternMatcher cannot be None"):
                RefactoringRanker(storage, None, fingerprinter)

            with pytest.raises(ValueError, match="PatternFingerprinter cannot be None"):
                RefactoringRanker(storage, matcher, None)

    def test_rank_operations_empty_list(self):
        """Test ranking empty operation list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            result = ranker.rank_operations([])
            assert result == []

    def test_rank_operations_without_patterns(self):
        """Test ranking operations when no patterns exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.3,
            )

            ranked = ranker.rank_operations([operation])
            assert len(ranked) == 1
            op, score = ranked[0]
            assert op == operation
            # Should have default score based on risk (1.0 - 0.3) * 0.8 = 0.56
            assert 0.0 <= score <= 1.0

    def test_rank_operations_with_matching_pattern(self):
        """Test ranking operations with matching patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            # Create a pattern
            pattern_hash = fingerprinter.fingerprint_code("def foo(): pass")
            pattern = RefactoringPattern.create(
                pattern_hash=pattern_hash,
                operation_type="extract_method",
                code_snippet_before="def foo(): pass",
                code_snippet_after="def bar(): pass",
            )
            pattern.acceptance_rate = 0.8
            pattern.total_occurrences = 10
            storage.save_pattern(pattern)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.2,
            )

            ranked = ranker.rank_operations([operation])
            assert len(ranked) == 1
            op, score = ranked[0]
            assert op == operation
            # Should have higher score due to matching pattern
            assert score > 0.5

    def test_rank_operations_sorts_by_score(self):
        """Test that ranked operations are sorted by score descending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            # Create high-acceptance pattern
            pattern_hash1 = fingerprinter.fingerprint_code("def high(): pass")
            pattern1 = RefactoringPattern.create(
                pattern_hash=pattern_hash1,
                operation_type="extract_method",
                code_snippet_before="def high(): pass",
                code_snippet_after="def high_refactored(): pass",
            )
            pattern1.acceptance_rate = 0.9
            pattern1.total_occurrences = 20
            storage.save_pattern(pattern1)

            # Create low-acceptance pattern
            pattern_hash2 = fingerprinter.fingerprint_code("def low(): pass")
            pattern2 = RefactoringPattern.create(
                pattern_hash=pattern_hash2,
                operation_type="extract_method",
                code_snippet_before="def low(): pass",
                code_snippet_after="def low_refactored(): pass",
            )
            pattern2.acceptance_rate = 0.3
            pattern2.total_occurrences = 5
            storage.save_pattern(pattern2)

            op1 = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="High acceptance",
                old_code="def high(): pass",
                new_code="def high_refactored(): pass",
                risk_score=0.2,
            )

            op2 = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=20,
                description="Low acceptance",
                old_code="def low(): pass",
                new_code="def low_refactored(): pass",
                risk_score=0.2,
            )

            ranked = ranker.rank_operations([op2, op1])  # Reverse order
            assert len(ranked) == 2
            # First should be op1 (higher score)
            assert ranked[0][0] == op1
            assert ranked[1][0] == op2
            # Scores should be descending
            assert ranked[0][1] >= ranked[1][1]

    def test_get_top_suggestions(self):
        """Test getting top N suggestions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            operations = [
                RefactoringOperation(
                    operation_type="extract_method",
                    file_path=Path("test.py"),
                    line_number=i,
                    description=f"Operation {i}",
                    old_code=f"code_{i}",
                    new_code=f"new_code_{i}",
                    risk_score=0.1 + (i * 0.1),  # Increasing risk
                )
                for i in range(5)
            ]

            top_3 = ranker.get_top_suggestions(operations, top_n=3)
            assert len(top_3) == 3
            # Should be sorted by score (which considers risk)
            assert all(isinstance(op, RefactoringOperation) for op in top_3)

    def test_get_top_suggestions_invalid_n(self):
        """Test get_top_suggestions with invalid top_n."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            with pytest.raises(ValueError, match="top_n must be positive"):
                ranker.get_top_suggestions([], top_n=0)

            with pytest.raises(ValueError, match="top_n must be positive"):
                ranker.get_top_suggestions([], top_n=-1)

    def test_rank_operations_applies_risk_penalty(self):
        """Test that risk penalty is applied to scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            # Create pattern
            pattern_hash = fingerprinter.fingerprint_code("def test(): pass")
            pattern = RefactoringPattern.create(
                pattern_hash=pattern_hash,
                operation_type="extract_method",
                code_snippet_before="def test(): pass",
                code_snippet_after="def test_refactored(): pass",
            )
            pattern.acceptance_rate = 0.8
            storage.save_pattern(pattern)

            # Low risk operation
            op_low_risk = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Low risk",
                old_code="def test(): pass",
                new_code="def test_refactored(): pass",
                risk_score=0.1,
            )

            # High risk operation
            op_high_risk = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=20,
                description="High risk",
                old_code="def test(): pass",
                new_code="def test_refactored(): pass",
                risk_score=0.9,
            )

            ranked = ranker.rank_operations([op_high_risk, op_low_risk])
            assert len(ranked) == 2

            # Low risk should rank higher
            assert ranked[0][0] == op_low_risk
            assert ranked[1][0] == op_high_risk
            assert ranked[0][1] > ranked[1][1]

    def test_rank_operations_with_project_profile(self):
        """Test ranking with project-specific profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            project_path = Path(tmpdir) / "project"
            project_path.mkdir()

            # Create pattern
            pattern_hash = fingerprinter.fingerprint_code("def test(): pass")
            pattern = RefactoringPattern.create(
                pattern_hash=pattern_hash,
                operation_type="extract_method",
                code_snippet_before="def test(): pass",
                code_snippet_after="def test_refactored(): pass",
            )
            pattern.acceptance_rate = 0.7
            storage.save_pattern(pattern)

            # Create project profile with custom weight
            profile = storage.get_project_profile(project_path)
            profile.set_pattern_weight(pattern.pattern_id, 0.9)  # High weight
            storage.save_project_profile(profile)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=project_path / "test.py",
                line_number=10,
                description="Test",
                old_code="def test(): pass",
                new_code="def test_refactored(): pass",
                risk_score=0.2,
            )

            ranked = ranker.rank_operations([operation], project_path=project_path)
            assert len(ranked) == 1
            score = ranked[0][1]
            # Score should be boosted by project weight
            assert score > 0.5

    def test_get_ranked_with_scores(self):
        """Test get_ranked_with_scores returns operations with scores and patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            # Create pattern
            pattern_hash = fingerprinter.fingerprint_code("def test(): pass")
            pattern = RefactoringPattern.create(
                pattern_hash=pattern_hash,
                operation_type="extract_method",
                code_snippet_before="def test(): pass",
                code_snippet_after="def test_refactored(): pass",
            )
            pattern.acceptance_rate = 0.8
            storage.save_pattern(pattern)

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Test",
                old_code="def test(): pass",
                new_code="def test_refactored(): pass",
                risk_score=0.2,
            )

            ranked = ranker.get_ranked_with_scores([operation])
            assert len(ranked) == 1
            op, score, matched_pattern = ranked[0]
            assert op == operation
            assert 0.0 <= score <= 1.0
            assert matched_pattern is not None
            assert matched_pattern.pattern_id == pattern.pattern_id

    def test_rank_operations_handles_fingerprint_failure(self, monkeypatch):
        """Test that ranking handles fingerprint failures gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PatternStorage(storage_dir=Path(tmpdir))
            matcher = PatternMatcher(storage)
            fingerprinter = PatternFingerprinter()
            ranker = RefactoringRanker(storage, matcher, fingerprinter)

            # Force fingerprinting to fail
            def _fail_fingerprint(_code: str) -> str:
                raise ValueError("fingerprint failure for testing")

            monkeypatch.setattr(ranker.fingerprinter, "fingerprint_code", _fail_fingerprint)

            # Operation that will trigger fingerprinting failure
            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=Path("test.py"),
                line_number=10,
                description="Test",
                old_code="def test(): pass",
                new_code="def new(): pass",
                risk_score=0.5,
            )

            # Should not raise, but assign default score
            ranked = ranker.rank_operations([operation])
            assert len(ranked) == 1
            op, score = ranked[0]
            assert op == operation
            assert 0.0 <= score <= 1.0
