"""Tests for the refactron repo CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from refactron.cli.repo import repo
from refactron.core.workspace import WorkspaceManager


@pytest.fixture
def runner():
    """Provides a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_workspace(tmp_path):
    """Provides an isolated workspace manager for tests."""
    config_path = tmp_path / "workspaces.json"
    mgr = WorkspaceManager(config_path=config_path)
    return mgr


@patch("refactron.cli.repo.WorkspaceManager")
@patch("refactron.cli.repo._spawn_background_indexer")
def test_repo_connect_local_offline(mock_spawn, mock_wsm_cls, runner, tmp_path):
    """Scenario 1 & 2: Inside existing local repo, connects offline instantly."""
    # Setup mock manager
    mock_mgr = MagicMock()
    mock_wsm_cls.return_value = mock_mgr

    # Mock detect_repository to simulate being inside a git repo
    mock_mgr.detect_repository.return_value = "user/my-offline-repo"

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(repo, ["connect"])

        # Verify it succeeds offline without hitting the API
        assert result.exit_code == 0
        assert "Connected (Local)" in result.output
        assert "detected from .git/config" in result.output
        assert "my-offline-repo" in result.output

        # Verify mapping was added
        mock_mgr.add_workspace.assert_called_once()
        mapping = mock_mgr.add_workspace.call_args[0][0]
        assert mapping.repo_full_name == "user/my-offline-repo"
        assert mapping.repo_name == "my-offline-repo"
        assert mapping.repo_id is None  # Offline = no ID

        # Verify background indexer was spawned
        mock_spawn.assert_called_once()


@patch("refactron.cli.repo.WorkspaceManager")
@patch("refactron.cli.repo.list_repositories")
@patch("refactron.cli.repo.subprocess.run")
@patch("refactron.cli.repo._spawn_background_indexer")
def test_repo_connect_api_fallback(
    mock_spawn, mock_subp_run, mock_list_repos, mock_wsm_cls, runner, tmp_path
):
    """Scenario 3: Outside git repo, repo name provided -> clones via API."""
    mock_mgr = MagicMock()
    mock_wsm_cls.return_value = mock_mgr

    # Simulate being OUTSIDE a git repo
    mock_mgr.detect_repository.return_value = None

    # Simulate API returning repositories
    mock_repo = MagicMock()
    mock_repo.name = "my-online-repo"
    mock_repo.full_name = "user/my-online-repo"
    mock_repo.id = 12345
    mock_repo.clone_url = "https://github.com/user/my-online-repo.git"
    mock_list_repos.return_value = [mock_repo]

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Pass repo explicit name
        result = runner.invoke(repo, ["connect", "user/my-online-repo"])

        assert result.exit_code == 0
        assert "Connected (API)" in result.output
        assert "cloned from GitHub" in result.output

        # Verify API list was called
        mock_list_repos.assert_called_once()

        # Verify git clone was triggered
        mock_subp_run.assert_called_once()
        assert "git" in mock_subp_run.call_args[0][0]
        assert "clone" in mock_subp_run.call_args[0][0]

        # Verify mapping was added with API ID
        mock_mgr.add_workspace.assert_called_once()
        mapping = mock_mgr.add_workspace.call_args[0][0]
        assert mapping.repo_id == 12345
        assert mapping.repo_name == "my-online-repo"


@patch("refactron.cli.repo.WorkspaceManager")
def test_repo_connect_outside_git_no_args(mock_wsm_cls, runner, tmp_path):
    """If outside git repo and no args provided, it should fail nicely."""
    mock_mgr = MagicMock()
    mock_wsm_cls.return_value = mock_mgr
    mock_mgr.detect_repository.return_value = None

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(repo, ["connect"])

        assert result.exit_code == 1
        assert "Not a git repository" in result.output
        assert "Usage" in result.output


@patch("refactron.cli.repo.WorkspaceManager")
@patch("refactron.cli.repo.list_repositories")
def test_repo_connect_api_error_fallback(mock_list_repos, mock_wsm_cls, runner, tmp_path):
    """Scenario 4: CI runner (no token, no git context) -> clean error message."""
    mock_mgr = MagicMock()
    mock_wsm_cls.return_value = mock_mgr
    mock_mgr.detect_repository.return_value = None

    # Simulate failure to authenticate to retrieve repositories
    mock_list_repos.side_effect = RuntimeError("Invalid credentials")

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(repo, ["connect", "some-repo"])

        assert result.exit_code == 1
        assert "Authentication required for cloning" in result.output
        assert "connect offline." in result.output.replace("\n", "")


@patch("refactron.cli.repo.WorkspaceManager")
@patch("refactron.cli.repo.list_repositories")
@patch("refactron.cli.repo.subprocess.run")
@patch("refactron.cli.repo._spawn_background_indexer")
def test_repo_connect_api_fallback_ssh(
    mock_spawn, mock_subp_run, mock_list_repos, mock_wsm_cls, runner, tmp_path
):
    """Scenario 6: Clone using SSH flag."""
    mock_mgr = MagicMock()
    mock_wsm_cls.return_value = mock_mgr
    mock_mgr.detect_repository.return_value = None

    mock_repo = MagicMock()
    mock_repo.name = "my-ssh-repo"
    mock_repo.full_name = "user/my-ssh-repo"
    mock_repo.id = 9999
    mock_repo.clone_url = "https://github.com/user/my-ssh-repo.git"
    mock_repo.ssh_url = "git@github.com:user/my-ssh-repo.git"
    mock_list_repos.return_value = [mock_repo]

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(repo, ["connect", "--ssh", "user/my-ssh-repo"])

        assert result.exit_code == 0
        assert "Connected (API)" in result.output
        assert "via SSH" in result.output

        mock_subp_run.assert_called_once()
        # Verify it used the ssh_url instead of clone_url
        assert "git@github.com:user/my-ssh-repo.git" in mock_subp_run.call_args[0][0]
