"""Tests for the backup and rollback system."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from refactron.core.backup import BackupManager, BackupRollbackSystem, GitIntegration
from refactron.core.credentials import RefactronCredentials


@pytest.fixture(autouse=True)
def mock_auth(monkeypatch):
    """Mock authentication for all CLI tests."""
    fake_creds = RefactronCredentials(
        api_base_url="https://api.refactron.dev",
        access_token="fake-token",
        token_type="Bearer",
        expires_at=None,
        email="test@example.com",
        plan="pro",
        api_key="ref_FAKE",
    )
    monkeypatch.setattr("refactron.cli.load_credentials", lambda: fake_creds)


class TestBackupManager:
    """Test suite for BackupManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = Path(tempfile.mkdtemp())
        yield temp
        if temp.exists():
            shutil.rmtree(temp)

    @pytest.fixture
    def backup_manager(self, temp_dir):
        """Create a BackupManager instance."""
        return BackupManager(root_dir=temp_dir)

    def test_initialization(self, backup_manager, temp_dir):
        """Test BackupManager initializes correctly."""
        assert backup_manager.root_dir == temp_dir
        assert backup_manager.backup_dir == temp_dir / ".refactron-backup"
        assert isinstance(backup_manager._index, dict)

    def test_create_backup_session(self, backup_manager):
        """Test creating a backup session."""
        session_id = backup_manager.create_backup_session("test operation")

        assert session_id.startswith("session_")
        assert len(backup_manager._index["sessions"]) == 1
        assert backup_manager._index["sessions"][0]["description"] == "test operation"

    def test_backup_file(self, backup_manager, temp_dir):
        """Test backing up a single file."""
        test_file = temp_dir / "test.py"
        test_file.write_text("print('hello')")

        session_id = backup_manager.create_backup_session("backup test")
        backup_path = backup_manager.backup_file(test_file, session_id)

        assert backup_path.exists()
        assert backup_path.read_text() == "print('hello')"

    def test_backup_file_not_found(self, backup_manager, temp_dir):
        """Test backing up a nonexistent file raises error."""
        nonexistent = temp_dir / "nonexistent.py"
        session_id = backup_manager.create_backup_session("test")

        with pytest.raises(FileNotFoundError):
            backup_manager.backup_file(nonexistent, session_id)

    def test_backup_multiple_files(self, backup_manager, temp_dir):
        """Test backing up multiple files."""
        file1 = temp_dir / "file1.py"
        file2 = temp_dir / "file2.py"
        file1.write_text("content1")
        file2.write_text("content2")

        session_id = backup_manager.create_backup_session("multi backup")
        backup_paths, failed = backup_manager.backup_files([file1, file2], session_id)

        assert len(backup_paths) == 2
        assert len(failed) == 0
        assert all(p.exists() for p in backup_paths)

    def test_rollback_session(self, backup_manager, temp_dir):
        """Test rolling back a session."""
        test_file = temp_dir / "rollback.py"
        test_file.write_text("original content")

        session_id = backup_manager.create_backup_session("rollback test")
        backup_manager.backup_file(test_file, session_id)

        test_file.write_text("modified content")
        assert test_file.read_text() == "modified content"

        restored_count, failed = backup_manager.rollback_session(session_id)

        assert restored_count == 1
        assert len(failed) == 0
        assert test_file.read_text() == "original content"

    def test_rollback_latest_session(self, backup_manager, temp_dir):
        """Test rolling back the latest session."""
        test_file = temp_dir / "latest.py"
        test_file.write_text("original")

        session_id = backup_manager.create_backup_session("latest test")
        backup_manager.backup_file(test_file, session_id)
        test_file.write_text("modified")

        restored_count, failed = backup_manager.rollback_session()

        assert restored_count == 1
        assert len(failed) == 0
        assert test_file.read_text() == "original"

    def test_rollback_no_sessions(self, backup_manager):
        """Test rollback with no sessions returns 0."""
        restored_count, failed = backup_manager.rollback_session()
        assert restored_count == 0
        assert len(failed) == 0

    def test_list_sessions(self, backup_manager):
        """Test listing backup sessions."""
        backup_manager.create_backup_session("session 1")
        backup_manager.create_backup_session("session 2")

        sessions = backup_manager.list_sessions()

        assert len(sessions) == 2
        assert sessions[0]["description"] == "session 1"
        assert sessions[1]["description"] == "session 2"

    def test_get_session(self, backup_manager):
        """Test getting a specific session."""
        session_id = backup_manager.create_backup_session("specific session")

        session = backup_manager.get_session(session_id)

        assert session is not None
        assert session["id"] == session_id
        assert session["description"] == "specific session"

    def test_get_nonexistent_session(self, backup_manager):
        """Test getting a nonexistent session returns None."""
        session = backup_manager.get_session("nonexistent_session")
        assert session is None

    def test_clear_session(self, backup_manager, temp_dir):
        """Test clearing a specific session."""
        test_file = temp_dir / "clear.py"
        test_file.write_text("content")

        session_id = backup_manager.create_backup_session("clear test")
        backup_manager.backup_file(test_file, session_id)

        assert backup_manager.get_session(session_id) is not None

        result = backup_manager.clear_session(session_id)

        assert result is True
        assert backup_manager.get_session(session_id) is None

    def test_clear_all_sessions(self, backup_manager, temp_dir):
        """Test clearing all sessions."""
        test_file = temp_dir / "clear_all.py"
        test_file.write_text("content")

        session1 = backup_manager.create_backup_session("session 1")
        backup_manager.create_backup_session("session 2")
        backup_manager.backup_file(test_file, session1)

        count = backup_manager.clear_all_sessions()

        assert count == 2
        assert len(backup_manager.list_sessions()) == 0

    def test_update_session_git_commit(self, backup_manager):
        """Test updating git commit for a session."""
        session_id = backup_manager.create_backup_session("git test")

        result = backup_manager.update_session_git_commit(session_id, "abc123def456")

        assert result is True
        session = backup_manager.get_session(session_id)
        assert session["git_commit"] == "abc123def456"

    def test_update_session_git_commit_nonexistent(self, backup_manager):
        """Test updating git commit for nonexistent session returns False."""
        result = backup_manager.update_session_git_commit("nonexistent", "abc123")
        assert result is False

    def test_get_latest_session(self, backup_manager):
        """Test getting the latest session."""
        backup_manager.create_backup_session("first")
        backup_manager.create_backup_session("second")
        backup_manager.create_backup_session("third")

        latest = backup_manager.get_latest_session()

        assert latest is not None
        assert latest["description"] == "third"

    def test_get_latest_session_empty(self, backup_manager):
        """Test getting latest session when none exist."""
        latest = backup_manager.get_latest_session()
        assert latest is None

    def test_backup_files_returns_failed(self, backup_manager, temp_dir):
        """Test backup_files returns list of failed files."""
        existing = temp_dir / "existing.py"
        nonexistent = temp_dir / "nonexistent.py"
        existing.write_text("content")

        session_id = backup_manager.create_backup_session("test")
        success, failed = backup_manager.backup_files([existing, nonexistent], session_id)

        assert len(success) == 1
        assert len(failed) == 1
        assert failed[0] == nonexistent

    def test_rollback_session_returns_failed(self, backup_manager, temp_dir):
        """Test rollback_session returns list of files that failed to restore."""
        test_file = temp_dir / "test.py"
        test_file.write_text("original")

        session_id = backup_manager.create_backup_session("test")
        backup_manager.backup_file(test_file, session_id)

        backup_path = backup_manager.backup_dir / session_id / "test.py"
        backup_path.unlink()

        restored, failed = backup_manager.rollback_session(session_id)

        assert restored == 0
        assert len(failed) == 1


class TestGitIntegration:
    """Test suite for GitIntegration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = Path(tempfile.mkdtemp())
        yield temp
        if temp.exists():
            shutil.rmtree(temp)

    @pytest.fixture
    def git_integration(self, temp_dir):
        """Create a GitIntegration instance."""
        return GitIntegration(repo_path=temp_dir)

    def test_is_git_repo_false(self, git_integration):
        """Test is_git_repo returns False for non-repo."""
        assert git_integration.is_git_repo() is False

    def test_is_git_repo_true(self, temp_dir, git_integration):
        """Test is_git_repo returns True for actual repo."""
        (temp_dir / ".git").mkdir()
        assert git_integration.is_git_repo() is True

    def test_has_uncommitted_changes_no_repo(self, git_integration):
        """Test has_uncommitted_changes returns False for non-repo."""
        assert git_integration.has_uncommitted_changes() is False

    def test_get_current_branch_no_repo(self, git_integration):
        """Test get_current_branch returns None for non-repo."""
        assert git_integration.get_current_branch() is None

    def test_get_current_commit_no_repo(self, git_integration):
        """Test get_current_commit returns None for non-repo."""
        assert git_integration.get_current_commit() is None

    def test_create_pre_refactor_commit_no_repo(self, git_integration):
        """Test create_pre_refactor_commit returns None for non-repo."""
        assert git_integration.create_pre_refactor_commit() is None

    def test_is_valid_git_ref(self):
        """Test _is_valid_git_ref validation."""
        assert GitIntegration._is_valid_git_ref("abc123def456") is True
        assert GitIntegration._is_valid_git_ref("a" * 40) is True
        assert GitIntegration._is_valid_git_ref("main") is True
        assert GitIntegration._is_valid_git_ref("feature/test") is True
        assert GitIntegration._is_valid_git_ref("") is False
        assert GitIntegration._is_valid_git_ref("abc;rm -rf /") is False
        assert GitIntegration._is_valid_git_ref("abc|cat") is False
        assert GitIntegration._is_valid_git_ref("abc$var") is False
        assert GitIntegration._is_valid_git_ref("../../../etc/passwd") is False
        assert GitIntegration._is_valid_git_ref(".hidden") is False
        assert GitIntegration._is_valid_git_ref("path..with..dots") is False
        assert GitIntegration._is_valid_git_ref("has space") is False


class TestGitIntegrationWithRepo:
    """Test GitIntegration with actual Git repositories."""

    @pytest.fixture
    def git_repo(self):
        """Create a temporary Git repository."""
        temp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init"], cwd=temp, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=temp,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=temp, capture_output=True, check=True
        )
        test_file = temp / "test.py"
        test_file.write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=temp, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=temp, capture_output=True, check=True
        )
        yield temp
        if temp.exists():
            shutil.rmtree(temp)

    def test_is_git_repo_true(self, git_repo):
        """Test is_git_repo returns True for actual repo."""
        git = GitIntegration(repo_path=git_repo)
        assert git.is_git_repo() is True

    def test_get_current_branch(self, git_repo):
        """Test get_current_branch returns branch name."""
        git = GitIntegration(repo_path=git_repo)
        branch = git.get_current_branch()
        assert branch is not None
        assert isinstance(branch, str)

    def test_get_current_commit(self, git_repo):
        """Test get_current_commit returns commit hash."""
        git = GitIntegration(repo_path=git_repo)
        commit = git.get_current_commit()
        assert commit is not None
        assert len(commit) == 40

    def test_has_uncommitted_changes_false(self, git_repo):
        """Test has_uncommitted_changes returns False when clean."""
        git = GitIntegration(repo_path=git_repo)
        assert git.has_uncommitted_changes() is False

    def test_has_uncommitted_changes_true(self, git_repo):
        """Test has_uncommitted_changes returns True when dirty."""
        (git_repo / "new.py").write_text("new content")
        git = GitIntegration(repo_path=git_repo)
        assert git.has_uncommitted_changes() is True

    def test_create_pre_refactor_commit(self, git_repo):
        """Test create_pre_refactor_commit creates a commit."""
        (git_repo / "new.py").write_text("new content")
        git = GitIntegration(repo_path=git_repo)

        old_commit = git.get_current_commit()
        new_commit = git.create_pre_refactor_commit(
            message="test commit", files=[git_repo / "new.py"]
        )

        assert new_commit is not None
        assert new_commit != old_commit

    def test_create_pre_refactor_commit_no_changes(self, git_repo):
        """Test create_pre_refactor_commit returns current commit when clean."""
        git = GitIntegration(repo_path=git_repo)
        current = git.get_current_commit()
        result = git.create_pre_refactor_commit()
        assert result == current

    def test_git_rollback_to_commit(self, git_repo):
        """Test git_rollback_to_commit restores files."""
        git = GitIntegration(repo_path=git_repo)
        original_commit = git.get_current_commit()

        test_file = git_repo / "test.py"
        test_file.write_text("modified content")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Modify"], cwd=git_repo, capture_output=True)

        result = git.git_rollback_to_commit(original_commit)

        assert result is True
        assert test_file.read_text() == "print('hello')"

    def test_git_rollback_invalid_ref(self, git_repo):
        """Test git_rollback_to_commit rejects invalid refs."""
        git = GitIntegration(repo_path=git_repo)
        result = git.git_rollback_to_commit("abc;rm -rf /")
        assert result is False


class TestBackupRollbackSystem:
    """Test suite for BackupRollbackSystem."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = Path(tempfile.mkdtemp())
        yield temp
        if temp.exists():
            shutil.rmtree(temp)

    @pytest.fixture
    def system(self, temp_dir):
        """Create a BackupRollbackSystem instance."""
        return BackupRollbackSystem(root_dir=temp_dir)

    def test_initialization(self, system, temp_dir):
        """Test system initializes correctly."""
        assert system.root_dir == temp_dir
        assert system.backup_manager is not None
        assert system.git is not None

    def test_prepare_for_refactoring(self, system, temp_dir):
        """Test preparing for refactoring creates backups."""
        test_file = temp_dir / "prepare.py"
        test_file.write_text("original")

        session_id, failed_files = system.prepare_for_refactoring(
            files=[test_file],
            description="test refactoring",
            create_git_commit=False,
        )

        assert session_id.startswith("session_")
        assert len(failed_files) == 0
        session = system.backup_manager.get_session(session_id)
        assert session is not None
        assert len(session["files"]) == 1

    def test_rollback_with_backup(self, system, temp_dir):
        """Test rolling back using file backups."""
        test_file = temp_dir / "rollback.py"
        test_file.write_text("original")

        session_id, _ = system.prepare_for_refactoring(
            files=[test_file],
            description="rollback test",
            create_git_commit=False,
        )

        test_file.write_text("modified")

        result = system.rollback(session_id=session_id, use_git=False)

        assert result["success"] is True
        assert result["files_restored"] == 1
        assert test_file.read_text() == "original"

    def test_rollback_no_sessions(self, system):
        """Test rollback with no sessions."""
        result = system.rollback(use_git=False)

        assert result["success"] is False
        assert "No files to restore" in result["message"]

    def test_list_sessions(self, system, temp_dir):
        """Test listing sessions through the system."""
        test_file = temp_dir / "list.py"
        test_file.write_text("content")

        system.prepare_for_refactoring(
            files=[test_file],
            description="list test",
            create_git_commit=False,
        )

        sessions = system.list_sessions()

        assert len(sessions) == 1
        assert sessions[0]["description"] == "list test"

    def test_clear_all(self, system, temp_dir):
        """Test clearing all sessions."""
        test_file = temp_dir / "clear.py"
        test_file.write_text("content")

        system.prepare_for_refactoring(
            files=[test_file],
            description="clear test",
            create_git_commit=False,
        )

        count = system.clear_all()

        assert count == 1
        assert len(system.list_sessions()) == 0


class TestCLIRollback:
    """Test CLI rollback command."""

    def test_rollback_help(self):
        """Test rollback help command."""
        from click.testing import CliRunner

        from refactron.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["rollback", "--help"])

        assert result.exit_code == 0
        assert "Rollback refactoring changes" in result.output

    def test_rollback_list_empty(self):
        """Test rollback --list with no sessions."""
        from click.testing import CliRunner

        from refactron.cli import main

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["rollback", "--list"])

        assert result.exit_code == 0
        assert "No backup sessions found" in result.output

    def test_rollback_no_sessions(self):
        """Test rollback with no sessions."""
        from click.testing import CliRunner

        from refactron.cli import main

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["rollback"])

        assert result.exit_code == 0
        assert "No backup sessions found" in result.output

    def test_rollback_nonexistent_session(self):
        """Test rollback with nonexistent session."""
        from click.testing import CliRunner

        from refactron.cli import main

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ["rollback", "--session", "nonexistent"])

        assert result.exit_code == 0
        assert "No backup sessions found" in result.output
