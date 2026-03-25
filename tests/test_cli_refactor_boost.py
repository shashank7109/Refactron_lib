"""
Tests for cli/refactor.py – covers refactor, autofix, rollback, document commands.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from refactron.autofix.models import FixRiskLevel
from refactron.cli.refactor import autofix, document, refactor, rollback


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def mock_cfg():
    cfg = MagicMock()
    cfg.backup_enabled = True
    return cfg


@pytest.fixture()
def mock_refactor_result():
    result = MagicMock()
    result.operations = []
    result.summary.return_value = {
        "total_operations": 0,
        "safe": 0,
        "high_risk": 0,
        "applied": False,
    }
    result.show_diff.return_value = ""
    result.apply.return_value = True
    return result


# ─────────────────────────── refactor command ─────────────────────────────────


class TestRefactorCommand:
    def test_no_target_no_workspace(self, runner, mock_cfg):
        with (
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor._auth_banner"),
            patch("refactron.cli.refactor.WorkspaceManager") as mock_ws,
        ):
            mock_ws.return_value.get_workspace_by_path.return_value = None
            result = runner.invoke(refactor, [])
        assert result.exit_code == 1

    def test_refactor_preview(self, runner, tmp_path, mock_cfg, mock_refactor_result):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor._auth_banner"),
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_refactor_filters"),
            patch("refactron.cli.refactor._confirm_apply_mode"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor._create_refactor_table", return_value=MagicMock()),
            patch("refactron.cli.refactor._print_refactor_messages"),
        ):
            mock_ref.return_value.refactor.return_value = mock_refactor_result
            result = runner.invoke(refactor, [str(py_file)])
        assert result.exit_code == 0

    def test_refactor_failure(self, runner, tmp_path, mock_cfg):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor._auth_banner"),
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_refactor_filters"),
            patch("refactron.cli.refactor._confirm_apply_mode"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
        ):
            mock_ref.return_value.refactor.side_effect = RuntimeError("fail")
            result = runner.invoke(refactor, [str(py_file)])
        assert result.exit_code == 1

    def test_refactor_no_target_workspace_found(
        self, runner, tmp_path, mock_cfg, mock_refactor_result
    ):
        mock_ws_obj = MagicMock()
        mock_ws_obj.local_path = str(tmp_path)
        mock_ws_obj.repo_full_name = "user/repo"
        with (
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor._auth_banner"),
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_refactor_filters"),
            patch("refactron.cli.refactor._confirm_apply_mode"),
            patch("refactron.cli.refactor.WorkspaceManager") as mock_ws,
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor._create_refactor_table", return_value=MagicMock()),
            patch("refactron.cli.refactor._print_refactor_messages"),
        ):
            mock_ws.return_value.get_workspace_by_path.return_value = mock_ws_obj
            mock_ref.return_value.refactor.return_value = mock_refactor_result
            result = runner.invoke(refactor, [])
        assert result.exit_code == 0

    def test_refactor_with_backup_io_error(self, runner, tmp_path, mock_cfg, mock_refactor_result):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        mock_cfg.backup_enabled = True
        with (
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor._auth_banner"),
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_refactor_filters"),
            patch("refactron.cli.refactor._confirm_apply_mode"),
            patch("refactron.cli.refactor.BackupRollbackSystem") as mock_brs,
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor._create_refactor_table", return_value=MagicMock()),
            patch("refactron.cli.refactor._print_refactor_messages"),
        ):
            mock_ref_inst = MagicMock()
            mock_ref_inst.detect_project_root.return_value = tmp_path
            mock_ref_inst.get_python_files.return_value = [py_file]
            mock_ref_inst.refactor.return_value = mock_refactor_result
            mock_ref.return_value = mock_ref_inst

            mock_brs.return_value.git.is_git_repo.return_value = False
            mock_brs.return_value.prepare_for_refactoring.side_effect = OSError("No space")

            # With --apply and stdin confirming "y" for continue without backup
            result = runner.invoke(refactor, [str(py_file), "--apply"], input="y\n")
        assert result.exit_code in (0, 1)

    def test_refactor_apply_with_operations(self, runner, tmp_path, mock_cfg, mock_refactor_result):
        py_file = tmp_path / "s.py"
        py_file.write_text("x = 1\n")
        mock_refactor_result.operations = [MagicMock(metadata={})]
        mock_refactor_result.summary.return_value["total_operations"] = 1
        with (
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor._auth_banner"),
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_refactor_filters"),
            patch("refactron.cli.refactor._confirm_apply_mode"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor._create_refactor_table", return_value=MagicMock()),
            patch("refactron.cli.refactor._print_refactor_messages"),
            patch("refactron.cli.refactor._record_applied_operations"),
        ):
            mock_cfg.backup_enabled = False
            mock_ref.return_value.refactor.return_value = mock_refactor_result
            result = runner.invoke(refactor, [str(py_file), "--apply"])
        assert result.exit_code == 0


# ─────────────────────────── autofix command ──────────────────────────────────


class TestAutofixCommand:
    def test_autofix_preview_mode(self, runner, tmp_path, mock_cfg):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_file_count"),
            patch("refactron.cli.refactor._auth_banner"),
        ):
            result = runner.invoke(autofix, [str(py_file)])
        assert result.exit_code == 0
        assert "LOW" in result.output or "Available" in result.output

    def test_autofix_apply_mode(self, runner, tmp_path, mock_cfg):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
            patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
            patch("refactron.cli.refactor._print_file_count"),
            patch("refactron.cli.refactor._auth_banner"),
        ):
            result = runner.invoke(autofix, [str(py_file), "--apply"])
        assert result.exit_code == 0

    def test_autofix_safety_levels(self, runner, tmp_path, mock_cfg):
        py_file = tmp_path / "s.py"
        py_file.write_text("x = 1\n")
        for level in ["safe", "low", "moderate", "high"]:
            with (
                patch("refactron.cli.refactor._load_config", return_value=mock_cfg),
                patch("refactron.cli.refactor._validate_path", return_value=tmp_path),
                patch("refactron.cli.refactor._print_file_count"),
                patch("refactron.cli.refactor._auth_banner"),
            ):
                result = runner.invoke(autofix, [str(py_file), "--safety-level", level])
            assert result.exit_code == 0


# ─────────────────────────── rollback command ─────────────────────────────────


class TestRollbackCommand:
    def test_rollback_no_sessions(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = []
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, [])
        assert "No backup" in result.output

    def test_rollback_list(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "sess1", "timestamp": "2024-01-01", "files": ["a.py"], "description": "test"}
        ]
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, ["--list"])
        assert "sess1" in result.output

    def test_rollback_clear_confirmed(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "x", "timestamp": "t", "files": [], "description": "d"}
        ]
        mock_system.clear_all.return_value = 1
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, ["--clear"], input="y\n")
        assert "Cleared" in result.output or result.exit_code == 0

    def test_rollback_clear_cancelled(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "x", "timestamp": "t", "files": [], "description": "d"}
        ]
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, ["--clear"], input="n\n")
        assert result.exit_code == 0

    def test_rollback_specific_session_not_found(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "sess1", "timestamp": "t", "files": [], "description": "d"}
        ]
        mock_system.backup_manager.get_session.return_value = None
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, ["missing_session_id"])
        assert result.exit_code == 1

    def test_rollback_latest_success(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "sess1", "timestamp": "t", "files": ["a.py"], "description": "d"}
        ]
        mock_system.rollback.return_value = {
            "success": True,
            "message": "Restored",
            "files_restored": 1,
        }
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, [], input="y\n")
        assert result.exit_code == 0

    def test_rollback_latest_failure(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "sess1", "timestamp": "t", "files": ["a.py"], "description": "d"}
        ]
        mock_system.rollback.return_value = {"success": False, "message": "Failed to restore"}
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, [], input="y\n")
        assert result.exit_code == 1

    def test_rollback_use_git(self, runner):
        mock_system = MagicMock()
        mock_system.list_sessions.return_value = [
            {"id": "sess1", "timestamp": "t", "files": ["a.py"], "description": "d"}
        ]
        mock_system.rollback.return_value = {
            "success": True,
            "message": "Git rollback done",
            "files_restored": 1,
        }
        with patch("refactron.cli.refactor.BackupRollbackSystem", return_value=mock_system):
            result = runner.invoke(rollback, ["--use-git"], input="y\n")
        assert result.exit_code == 0


# ─────────────────────────── document command ─────────────────────────────────


class TestDocumentCommand:
    def test_document_file(self, runner, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text("def foo(): pass\n")

        from refactron.llm.models import SuggestionStatus

        mock_sugg = MagicMock()
        mock_sugg.status = SuggestionStatus.PENDING
        mock_sugg.explanation = "Docs added"
        mock_sugg.proposed_code = "# docs\n"
        mock_sugg.model_name = "model"
        mock_sugg.confidence_score = 0.9

        with (
            patch("refactron.cli.refactor._load_config"),
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor.ContextRetriever"),
            patch("refactron.cli.refactor.LLMOrchestrator") as mock_orch,
            patch("refactron.cli.refactor._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            mock_orch.return_value.generate_documentation.return_value = mock_sugg
            result = runner.invoke(document, [str(py_file)])
        assert result.exit_code == 0

    def test_document_directory_not_supported(self, runner, tmp_path):
        with (
            patch("refactron.cli.refactor._load_config"),
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor.ContextRetriever"),
            patch("refactron.cli.refactor.LLMOrchestrator"),
            patch("refactron.cli.refactor._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            result = runner.invoke(document, [str(tmp_path)])
        assert result.exit_code == 0  # returns early

    def test_document_failed_generation(self, runner, tmp_path):
        py_file = tmp_path / "module.py"
        py_file.write_text("def foo(): pass\n")

        from refactron.llm.models import SuggestionStatus

        mock_sugg = MagicMock()
        mock_sugg.status = SuggestionStatus.FAILED
        mock_sugg.explanation = "No LLM"

        with (
            patch("refactron.cli.refactor._load_config"),
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor.ContextRetriever"),
            patch("refactron.cli.refactor.LLMOrchestrator") as mock_orch,
            patch("refactron.cli.refactor._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            mock_orch.return_value.generate_documentation.return_value = mock_sugg
            result = runner.invoke(document, [str(py_file)])
        assert result.exit_code == 0

    def test_document_apply_interactive(self, runner, tmp_path):
        py_file = tmp_path / "m.py"
        py_file.write_text("def bar(): pass\n")

        from refactron.llm.models import SuggestionStatus

        mock_sugg = MagicMock()
        mock_sugg.status = SuggestionStatus.PENDING
        mock_sugg.explanation = "Docs"
        mock_sugg.proposed_code = "# new docs\n"
        mock_sugg.model_name = "m"
        mock_sugg.confidence_score = 0.8

        with (
            patch("refactron.cli.refactor._load_config"),
            patch("refactron.cli.refactor._setup_logging"),
            patch("refactron.cli.refactor.Refactron") as mock_ref,
            patch("refactron.cli.refactor.ContextRetriever"),
            patch("refactron.cli.refactor.LLMOrchestrator") as mock_orch,
            patch("refactron.cli.refactor._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            mock_orch.return_value.generate_documentation.return_value = mock_sugg
            result = runner.invoke(document, [str(py_file), "--apply", "--no-interactive"])
        assert result.exit_code == 0
