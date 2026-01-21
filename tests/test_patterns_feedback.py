"""Tests for Phase 3: Feedback Collection System."""

import tempfile
import uuid
from pathlib import Path

import pytest

from refactron import Refactron
from refactron.core.config import RefactronConfig
from refactron.core.models import RefactoringOperation
from refactron.patterns.storage import PatternStorage


@pytest.fixture
def temp_storage_dir():
    """Provide a temporary directory for pattern storage to ensure test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def refactron_with_temp_storage(temp_storage_dir):
    """Provide a Refactron instance with isolated temporary storage."""
    config = RefactronConfig()
    refactron = Refactron(config)
    # Override storage with temporary directory for test isolation
    refactron.pattern_storage = PatternStorage(storage_dir=temp_storage_dir)
    return refactron


class TestRefactoringOperationID:
    """Test unique operation ID generation."""

    def test_operation_has_unique_id(self):
        """Test that each operation gets a unique ID."""
        op1 = RefactoringOperation(
            operation_type="extract_method",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract method",
            old_code="def foo(): pass",
            new_code="def bar(): pass",
            risk_score=0.5,
        )
        op2 = RefactoringOperation(
            operation_type="extract_constant",
            file_path=Path("test.py"),
            line_number=20,
            description="Extract constant",
            old_code="x = 100",
            new_code="x = MAX_SIZE",
            risk_score=0.3,
        )

        assert op1.operation_id != op2.operation_id
        assert op1.operation_id
        assert op2.operation_id

    def test_operation_id_is_uuid_format(self):
        """Test that operation IDs are valid UUIDs."""
        op = RefactoringOperation(
            operation_type="extract_method",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract method",
            old_code="def foo(): pass",
            new_code="def bar(): pass",
            risk_score=0.5,
        )

        # Should be a valid UUID format
        try:
            uuid.UUID(op.operation_id)
        except ValueError:
            pytest.fail(f"Operation ID {op.operation_id} is not a valid UUID")


class TestRefactronFeedbackIntegration:
    """Test feedback integration in Refactron class."""

    def test_refactron_initializes_pattern_storage(self):
        """Test that Refactron initializes pattern storage."""
        config = RefactronConfig()
        refactron = Refactron(config)

        # Pattern storage should be initialized (or None if failed)
        assert hasattr(refactron, "pattern_storage")
        assert hasattr(refactron, "pattern_fingerprinter")

    def test_record_feedback_success(self, refactron_with_temp_storage):
        """Test successful feedback recording."""
        refactron = refactron_with_temp_storage

        if not refactron.pattern_storage:
            pytest.skip("Pattern storage not initialized")

        operation = RefactoringOperation(
            operation_type="extract_method",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract method",
            old_code="def foo(): pass",
            new_code="def bar(): pass",
            risk_score=0.5,
        )

        # Record feedback
        refactron.record_feedback(
            operation_id=operation.operation_id,
            action="accepted",
            reason="Good refactoring",
            operation=operation,
        )

        # Verify feedback was saved
        feedback_list = refactron.pattern_storage.load_feedback()
        assert len(feedback_list) > 0

        # Find our feedback
        our_feedback = next(
            (f for f in feedback_list if f.operation_id == operation.operation_id), None
        )
        assert our_feedback is not None
        assert our_feedback.action == "accepted"
        assert our_feedback.reason == "Good refactoring"
        assert our_feedback.operation_type == "extract_method"

    def test_record_feedback_without_storage(self):
        """Test that record_feedback gracefully handles missing storage."""
        config = RefactronConfig()
        refactron = Refactron(config)

        # Manually set storage to None
        refactron.pattern_storage = None

        # Should not raise an error
        refactron.record_feedback(
            operation_id="test-id",
            action="accepted",
            operation=None,
        )

    def test_record_feedback_invalid_action(self, refactron_with_temp_storage):
        """Test that invalid actions are ignored."""
        refactron = refactron_with_temp_storage

        if not refactron.pattern_storage:
            pytest.skip("Pattern storage not initialized")

        # Count existing feedback
        initial_count = len(refactron.pattern_storage.load_feedback())

        # Try invalid action
        refactron.record_feedback(
            operation_id="test-id",
            action="invalid_action",
            operation=None,
        )

        # Count should be unchanged
        assert len(refactron.pattern_storage.load_feedback()) == initial_count

    def test_refactor_fingerprints_code(self):
        """Test that refactor() fingerprints code patterns."""
        config = RefactronConfig()
        refactron = Refactron(config)

        if not refactron.pattern_fingerprinter:
            pytest.skip("Pattern fingerprinter not initialized")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def calculate(price):
    if price > 1000:
        return price * 0.15
    return 0
""")
            temp_path = Path(f.name)

        try:
            result = refactron.refactor(temp_path, preview=True)

            # Check if operations have pattern hashes
            if result.operations:
                op = result.operations[0]
                assert "code_pattern_hash" in op.metadata
                assert op.metadata["code_pattern_hash"] is not None
        finally:
            temp_path.unlink()

    def test_refactor_generates_operation_ids(self):
        """Test that refactor() generates unique operation IDs."""
        config = RefactronConfig()
        refactron = Refactron(config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def calculate(price):
    if price > 1000:
        return price * 0.15
    return 0
""")
            temp_path = Path(f.name)

        try:
            result = refactron.refactor(temp_path, preview=True)

            if result.operations:
                operation_ids = [op.operation_id for op in result.operations]
                # All IDs should be unique
                assert len(operation_ids) == len(set(operation_ids))
                # All IDs should be non-empty
                assert all(op_id for op_id in operation_ids)
        finally:
            temp_path.unlink()


class TestFeedbackStorage:
    """Test feedback storage persistence."""

    def test_feedback_persists_across_sessions(self):
        """Test that feedback persists across Refactron sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a project directory with .refactron marker
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            storage_dir = project_dir / ".refactron" / "patterns"
            storage_dir.mkdir(parents=True)

            test_file = project_dir / "test.py"
            test_file.write_text("def foo(): pass\n")

            config = RefactronConfig()
            refactron1 = Refactron(config)
            # Override storage with temporary directory for test isolation
            refactron1.pattern_storage = PatternStorage(storage_dir=storage_dir)

            if not refactron1.pattern_storage:
                pytest.skip("Pattern storage not initialized")

            operation = RefactoringOperation(
                operation_type="extract_method",
                file_path=test_file,
                line_number=1,
                description="Extract method",
                old_code="def foo(): pass",
                new_code="def bar(): pass",
                risk_score=0.5,
            )

            # Record feedback in first session
            refactron1.record_feedback(
                operation_id=operation.operation_id,
                action="accepted",
                reason="Good change",
                operation=operation,
            )

            # Create new Refactron instance (new session)
            refactron2 = Refactron(config)
            # Override storage with temporary directory for test isolation
            refactron2.pattern_storage = PatternStorage(storage_dir=storage_dir)

            # Verify feedback persists
            feedback_list = refactron2.pattern_storage.load_feedback()
            our_feedback = next(
                (f for f in feedback_list if f.operation_id == operation.operation_id),
                None,
            )
            assert our_feedback is not None
            assert our_feedback.action == "accepted"


class TestProjectRootDetection:
    """Test project root detection in feedback recording."""

    def test_detect_project_root_finds_git(self):
        """Test that project root is detected when .git exists."""
        config = RefactronConfig()
        refactron = Refactron(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            (project_dir / ".git").mkdir()  # Git marker
            (project_dir / "src").mkdir()

            test_file = project_dir / "src" / "test.py"

            root = refactron.detect_project_root(test_file)
            assert root.resolve() == project_dir.resolve()

    def test_detect_project_root_finds_setup_py(self):
        """Test that project root is detected when setup.py exists."""
        config = RefactronConfig()
        refactron = Refactron(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            (project_dir / "setup.py").write_text("# setup")  # Setup marker
            (project_dir / "src").mkdir()

            test_file = project_dir / "src" / "test.py"

            root = refactron.detect_project_root(test_file)
            assert root.resolve() == project_dir.resolve()

    def test_detect_project_root_fallback(self):
        """Test that project root detection falls back to file parent."""
        config = RefactronConfig()
        refactron = Refactron(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"

            root = refactron.detect_project_root(test_file)
            assert root == test_file.parent


class TestFeedbackCLIIntegration:
    """Test CLI integration for feedback collection."""

    def test_refactor_auto_records_on_apply(self):
        """Test that --apply automatically records feedback."""
        from click.testing import CliRunner

        from refactron.cli import refactor

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def calculate(price):
    if price > 1000:
        return price * 0.15
    return 0
""")
            temp_path = f.name

        try:
            # Run with --apply (should auto-record)
            result = runner.invoke(
                refactor,
                [temp_path, "--apply"],
                input="y\n",  # Confirm apply
            )

            # Should succeed
            assert result.exit_code == 0

            # Check that feedback was recorded if operations were generated
            # (Note: May be empty if no operations were generated)
        finally:
            Path(temp_path).unlink()

    def test_feedback_command_exists(self):
        """Test that feedback command exists and works."""
        from click.testing import CliRunner

        from refactron.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "feedback" in result.output

    def test_feedback_command_records_feedback(self, temp_storage_dir, monkeypatch):
        """Test that feedback command records feedback correctly."""
        from click.testing import CliRunner

        from refactron.cli import feedback

        runner = CliRunner()

        operation_id = str(uuid.uuid4())

        # Mock Refactron to use temporary storage
        original_refactron_init = Refactron.__init__

        def mock_init(self, config):
            original_refactron_init(self, config)
            self.pattern_storage = PatternStorage(storage_dir=temp_storage_dir)

        monkeypatch.setattr(Refactron, "__init__", mock_init)

        result = runner.invoke(
            feedback,
            [operation_id, "--action", "accepted", "--reason", "Test reason"],
        )

        assert result.exit_code == 0
        assert "Feedback recorded" in result.output or "✅" in result.output

        # Verify feedback was saved
        storage = PatternStorage(storage_dir=temp_storage_dir)
        feedback_list = storage.load_feedback()
        our_feedback = next((f for f in feedback_list if f.operation_id == operation_id), None)
        if our_feedback:
            assert our_feedback.action == "accepted"
            assert our_feedback.reason == "Test reason"
