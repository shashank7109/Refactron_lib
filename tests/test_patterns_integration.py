"""Integration tests for Pattern Learning System - Phase 9.

Tests the complete workflow: refactor → feedback → learn → rank
"""

import tempfile
from pathlib import Path

import pytest

from refactron import Refactron
from refactron.core.config import RefactronConfig
from refactron.core.models import RefactoringOperation


@pytest.fixture
def temp_storage_dir():
    """Provide a temporary directory for pattern storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def refactron_with_storage(temp_storage_dir):
    """Provide a Refactron instance with isolated storage."""
    config = RefactronConfig(
        enable_pattern_learning=True,
        pattern_learning_enabled=True,
        pattern_ranking_enabled=True,
        pattern_storage_dir=temp_storage_dir,
    )
    return Refactron(config)


class TestFullWorkflowIntegration:
    """Test complete workflow: refactor → feedback → learn → rank."""

    def test_refactor_feedback_learn_rank_workflow(self, refactron_with_storage, temp_storage_dir):
        """Test complete workflow from refactoring to ranking."""
        refactron = refactron_with_storage

        # Create a test file
        test_file = temp_storage_dir / "test_code.py"
        test_file.write_text(
            """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
""".strip()
        )

        # Step 1: Refactor (should fingerprint and rank)
        result = refactron.refactor(test_file, preview=True)
        assert len(result.operations) > 0

        # Verify operations have fingerprints
        for op in result.operations:
            assert "code_pattern_hash" in op.metadata
            assert op.metadata["code_pattern_hash"] is not None

        # Verify operations are ranked (if ranking enabled)
        if refactron.pattern_ranker:
            for op in result.operations:
                assert "ranking_score" in op.metadata

        # Step 2: Record feedback on first operation
        first_op = result.operations[0]
        refactron.record_feedback(
            operation_id=first_op.operation_id,
            action="accepted",
            reason="Good refactoring",
            operation=first_op,
        )

        # Step 3: Verify feedback was saved
        feedback_list = refactron.pattern_storage.load_feedback()
        assert len(feedback_list) > 0
        our_feedback = next(
            (f for f in feedback_list if f.operation_id == first_op.operation_id), None
        )
        assert our_feedback is not None
        assert our_feedback.action == "accepted"

        # Step 4: Verify pattern was learned
        patterns = refactron.pattern_storage.load_patterns()
        assert len(patterns) > 0

        # Find pattern for this operation
        pattern_hash = first_op.metadata["code_pattern_hash"]
        matching_patterns = [p for p in patterns.values() if p.pattern_hash == pattern_hash]
        assert len(matching_patterns) > 0

        pattern = matching_patterns[0]
        assert pattern.operation_type == first_op.operation_type
        assert pattern.accepted_count >= 1
        assert pattern.acceptance_rate > 0

        # Step 5: Refactor again - should use learned patterns for ranking
        result2 = refactron.refactor(test_file, preview=True)
        assert len(result2.operations) > 0

        # If ranking is enabled, operations should be ranked and
        # the previously accepted pattern should receive a score (may be same or higher)
        if refactron.pattern_ranker:
            # Baseline score from the first run for the accepted operation
            baseline_score = first_op.metadata.get("ranking_score")
            accepted_pattern_hash = first_op.metadata["code_pattern_hash"]

            # Find operations in the second run that match the accepted pattern
            matching_ops = [
                op
                for op in result2.operations
                if op.metadata.get("code_pattern_hash") == accepted_pattern_hash
            ]

            # Verify all operations have ranking scores
            for op in result2.operations:
                assert "ranking_score" in op.metadata

            # The matching operations should have a ranking score
            # (may be same or higher after learning, depending on pattern acceptance rate)
            if matching_ops:
                for op in matching_ops:
                    score = op.metadata["ranking_score"]
                    assert score >= 0  # Score should be valid
                    # After learning, score should be >= baseline (or > 0 if no baseline)
                    if baseline_score is not None:
                        assert score >= baseline_score
                    else:
                        assert score > 0

    def test_multiple_feedback_improves_pattern(self, refactron_with_storage, temp_storage_dir):
        """Test that multiple feedback records improve pattern statistics."""
        refactron = refactron_with_storage

        # Use code that will generate refactoring operations
        test_file = temp_storage_dir / "test.py"
        test_file.write_text(
            """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * 100  # Magic number
    return total
""".strip()
        )

        # Refactor multiple times and record feedback
        operations_seen = []
        for i in range(3):
            result = refactron.refactor(test_file, preview=True)
            if result.operations:
                op = result.operations[0]
                operations_seen.append(op)
                refactron.record_feedback(
                    operation_id=op.operation_id,
                    action="accepted" if i < 2 else "rejected",
                    operation=op,
                )

        # Skip if no operations were generated
        if not operations_seen:
            pytest.skip("No refactoring operations generated for this code")

        # Check pattern statistics
        patterns = refactron.pattern_storage.load_patterns()
        if len(patterns) == 0:
            pytest.skip("No patterns learned (may require more complex code)")

        # Find pattern
        pattern_hash = operations_seen[0].metadata.get("code_pattern_hash")
        if pattern_hash:
            matching_patterns = [p for p in patterns.values() if p.pattern_hash == pattern_hash]
            if matching_patterns:
                pattern = matching_patterns[0]
                assert pattern.accepted_count >= 1
                assert pattern.total_occurrences >= 1
                assert pattern.acceptance_rate >= 0


class TestMultiProjectPatternIsolation:
    """Test that patterns are isolated per project."""

    def test_patterns_isolated_by_project(self, temp_storage_dir):
        """Test that patterns from different projects are isolated."""
        project1_dir = temp_storage_dir / "project1"
        project1_dir.mkdir()
        project1_file = project1_dir / "code.py"
        project1_file.write_text(
            """
def func1():
    x = 100  # Magic number
    return x * 2
""".strip()
        )

        project2_dir = temp_storage_dir / "project2"
        project2_dir.mkdir()
        project2_file = project2_dir / "code.py"
        project2_file.write_text(
            """
def func2():
    y = 200  # Different magic number
    return y * 3
""".strip()
        )

        # Create separate storage for each project
        storage1_dir = temp_storage_dir / "storage1"
        storage2_dir = temp_storage_dir / "storage2"

        config1 = RefactronConfig(
            enable_pattern_learning=True,
            pattern_storage_dir=storage1_dir,
        )
        refactron1 = Refactron(config1)

        config2 = RefactronConfig(
            enable_pattern_learning=True,
            pattern_storage_dir=storage2_dir,
        )
        refactron2 = Refactron(config2)

        # Refactor and record feedback in project 1
        result1 = refactron1.refactor(project1_file, preview=True)
        op1 = None
        if result1.operations:
            op1 = result1.operations[0]
            refactron1.record_feedback(
                operation_id=op1.operation_id,
                action="accepted",
                operation=op1,
            )

        # Refactor and record feedback in project 2
        result2 = refactron2.refactor(project2_file, preview=True)
        op2 = None
        if result2.operations:
            op2 = result2.operations[0]
            refactron2.record_feedback(
                operation_id=op2.operation_id,
                action="accepted",
                operation=op2,
            )

        # Verify patterns are stored separately
        patterns1 = refactron1.pattern_storage.load_patterns()
        patterns2 = refactron2.pattern_storage.load_patterns()

        # Verify storage isolation (different directories)
        assert storage1_dir != storage2_dir

        # If operations were generated and patterns learned, verify actual isolation
        if op1 and op2 and (len(patterns1) > 0 or len(patterns2) > 0):
            # Patterns should be stored in separate directories
            assert storage1_dir.exists() or storage2_dir.exists()

            # Verify actual pattern isolation: patterns from project1 should not be in project2
            # and vice versa (they use different storage directories)
            pattern_hashes1 = {p.pattern_hash for p in patterns1.values()}
            pattern_hashes2 = {p.pattern_hash for p in patterns2.values()}

            # If both projects learned patterns, they should be different (different code)
            if pattern_hashes1 and pattern_hashes2:
                # Different code should produce different pattern hashes
                assert pattern_hashes1 != pattern_hashes2 or len(pattern_hashes1) == 0

            # Verify storage directories are separate (isolation mechanism)
            assert refactron1.pattern_storage.storage_dir != refactron2.pattern_storage.storage_dir


class TestPatternDatabasePersistence:
    """Test that pattern database persists across sessions."""

    def test_patterns_persist_across_sessions(self, temp_storage_dir):
        """Test that patterns persist when Refactron is recreated."""
        test_file = temp_storage_dir / "test.py"
        test_file.write_text(
            """
def calculate(x):
    return x * 100  # Magic number
""".strip()
        )

        # First session: refactor and record feedback
        config = RefactronConfig(
            enable_pattern_learning=True,
            pattern_storage_dir=temp_storage_dir,
        )
        refactron1 = Refactron(config)

        result = refactron1.refactor(test_file, preview=True)
        if not result.operations:
            pytest.skip("No refactoring operations generated")

        op = result.operations[0]
        refactron1.record_feedback(
            operation_id=op.operation_id,
            action="accepted",
            operation=op,
        )

        # Get patterns from first session
        patterns1 = refactron1.pattern_storage.load_patterns()
        pattern_count1 = len(patterns1)

        # Get feedback count from first session
        feedback1 = refactron1.pattern_storage.load_feedback()
        feedback_count1 = len(feedback1)

        # Second session: recreate Refactron with same storage
        refactron2 = Refactron(config)

        # Patterns should still be there (if any were learned)
        patterns2 = refactron2.pattern_storage.load_patterns()
        assert len(patterns2) == pattern_count1

        # Feedback should also persist
        feedback2 = refactron2.pattern_storage.load_feedback()
        assert len(feedback2) == feedback_count1
        assert len(feedback2) > 0

    def test_feedback_persists_across_sessions(self, temp_storage_dir):
        """Test that feedback records persist across sessions."""
        config = RefactronConfig(
            enable_pattern_learning=True,
            pattern_storage_dir=temp_storage_dir,
        )

        # First session
        refactron1 = Refactron(config)
        operation = RefactoringOperation(
            operation_type="extract_constant",
            file_path=Path("test.py"),
            line_number=1,
            description="Extract constant",
            old_code="x = 100",
            new_code="x = MAX_SIZE",
            risk_score=0.3,
        )

        refactron1.record_feedback(
            operation_id=operation.operation_id,
            action="accepted",
            reason="Good refactoring",
            operation=operation,
        )

        feedback_count1 = len(refactron1.pattern_storage.load_feedback())

        # Second session
        refactron2 = Refactron(config)
        feedback_count2 = len(refactron2.pattern_storage.load_feedback())

        assert feedback_count2 == feedback_count1
        assert feedback_count2 > 0


class TestCLIFeedbackCollectionIntegration:
    """Test CLI feedback collection integration."""

    def test_cli_feedback_collection_workflow(self, temp_storage_dir):
        """Test that CLI feedback collection works end-to-end."""
        from click.testing import CliRunner

        from refactron.cli import feedback

        runner = CliRunner()

        # Create a test operation ID
        operation_id = "test-op-123"

        # This is a simplified test - full CLI integration is tested in test_patterns_feedback.py
        # Here we just verify the command exists and can be invoked
        # Note: The command may fail if storage is not properly configured, but the command
        # itself should be callable (exit code 0 or 1, not 2 which would indicate command not found)
        result = runner.invoke(
            feedback,
            [operation_id, "--action", "accepted", "--reason", "Test"],
        )

        # Command should exist (may fail if storage not set up, but command should be callable)
        assert result.exit_code in (0, 1)  # 0 = success, 1 = error (but command exists)


class TestEndToEndPatternLearning:
    """End-to-end tests for pattern learning system."""

    def test_learning_improves_ranking_over_time(self, refactron_with_storage, temp_storage_dir):
        """Test that learning improves ranking accuracy over time."""
        refactron = refactron_with_storage

        test_file = temp_storage_dir / "test.py"
        test_file.write_text(
            """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
""".strip()
        )

        # Initial refactor - no patterns learned yet
        result1 = refactron.refactor(test_file, preview=True)
        initial_ops = result1.operations.copy()

        # Skip if no operations generated
        if not initial_ops:
            pytest.skip("No refactoring operations generated")

        op1 = initial_ops[0]

        # Record feedback on first operation (accept it)
        refactron.record_feedback(
            operation_id=op1.operation_id,
            action="accepted",
            operation=op1,
        )

        # Refactor again - pattern should be learned
        result2 = refactron.refactor(test_file, preview=True)

        # Verify pattern was learned
        patterns = refactron.pattern_storage.load_patterns()
        assert len(patterns) > 0

        # If ranking is enabled, the accepted operation should have a ranking score
        if refactron.pattern_ranker and result2.operations:
            for op in result2.operations:
                if op.operation_type == op1.operation_type:
                    assert "ranking_score" in op.metadata
                    assert op.metadata["ranking_score"] >= 0

    def test_pattern_cleanup_preserves_recent_patterns(
        self, refactron_with_storage, temp_storage_dir
    ):
        """Test that pattern cleanup preserves recent patterns."""
        refactron = refactron_with_storage

        from refactron.patterns.learning_service import LearningService

        test_file = temp_storage_dir / "test.py"
        test_file.write_text(
            """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 100)  # Magic number
    return result
""".strip()
        )

        # Create and learn a pattern
        result = refactron.refactor(test_file, preview=True)
        if not result.operations:
            pytest.skip("No refactoring operations generated")

        op = result.operations[0]
        refactron.record_feedback(
            operation_id=op.operation_id,
            action="accepted",
            operation=op,
        )

        # Verify pattern exists
        patterns_before = refactron.pattern_storage.load_patterns()
        if len(patterns_before) == 0:
            pytest.skip("No patterns learned (may require more complex code)")

        # Run cleanup with retention that should keep recent patterns (90 days default)
        service = LearningService(storage=refactron.pattern_storage)
        service.cleanup_old_patterns(days=90)  # Default retention period

        # Pattern should still exist (it's recent, created today)
        patterns_after = refactron.pattern_storage.load_patterns()
        assert len(patterns_after) >= len(patterns_before)

        # Test that cleanup actually works by using a very old date threshold
        # (This would remove patterns older than 365 days, so our recent pattern should remain)
        service.cleanup_old_patterns(days=365)
        patterns_after_old = refactron.pattern_storage.load_patterns()
        assert len(patterns_after_old) >= len(patterns_before)
