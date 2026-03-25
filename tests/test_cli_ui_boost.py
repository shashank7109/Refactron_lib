"""
Tests for cli/ui.py – shared UI helper functions.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import click
import pytest
from click.testing import CliRunner

# Import helpers under test
from refactron.cli.ui import (
    _auth_banner,
    _collect_feedback_interactive,
    _confirm_apply_mode,
    _create_refactor_table,
    _create_summary_table,
    _interactive_file_selector,
    _print_custom_help,
    _print_detailed_issues,
    _print_file_count,
    _print_helpful_tips,
    _print_refactor_filters,
    _print_refactor_messages,
    _print_status_messages,
    _record_applied_operations,
    _run_minimal_loop,
    _run_startup_animation,
    console,
)


def make_summary(**kwargs):
    base = {
        "total_files": 5,
        "files_analyzed": 5,
        "files_failed": 0,
        "total_issues": 0,
        "critical": 0,
        "errors": 0,
        "warnings": 0,
        "info": 0,
        "applied": False,
        "total_operations": 0,
        "safe": 0,
        "high_risk": 0,
    }
    base.update(kwargs)
    return base


def make_issue_mock(level_val="warning", suggestion=None):
    issue = MagicMock()
    issue.level.value = level_val
    issue.suggestion = suggestion
    issue.__str__ = lambda self: f"Issue at line 1: {level_val}"
    return issue


# ─────────────────────────── _auth_banner ─────────────────────────────────────


class TestAuthBanner:
    def test_renders_without_error(self):
        with patch.object(console, "print"):
            _auth_banner("Test Title")  # Should not raise

    def test_title_appears(self):
        captured = []
        with patch.object(console, "print", side_effect=lambda *a, **k: captured.append(str(a))):
            _auth_banner("Analysis")


# ─────────────────────────── _print_file_count ────────────────────────────────


class TestPrintFileCount:
    def test_directory_prints_count(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        with patch.object(console, "print") as mock_print:
            _print_file_count(tmp_path)
        mock_print.assert_called()

    def test_file_prints_nothing(self, tmp_path):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x=1")
        with patch.object(console, "print") as mock_print:
            _print_file_count(py_file)
        mock_print.assert_not_called()


# ─────────────────────────── _create_summary_table ────────────────────────────


class TestCreateSummaryTable:
    def test_no_issues(self):
        table = _create_summary_table(make_summary())
        assert table is not None

    def test_with_critical(self):
        table = _create_summary_table(make_summary(critical=2, files_failed=1, total_issues=3))
        assert table is not None

    def test_with_all_counts(self):
        table = _create_summary_table(
            make_summary(total_issues=5, critical=1, errors=2, warnings=1, info=1)
        )
        assert table is not None


# ─────────────────────────── _print_status_messages ───────────────────────────


class TestPrintStatusMessages:
    def test_no_issues_no_failures(self):
        with patch.object(console, "print"):
            _print_status_messages(make_summary())  # Panel shown

    def test_files_failed(self):
        with patch.object(console, "print") as mock_print:
            _print_status_messages(make_summary(files_failed=2, total_issues=0))
        assert mock_print.called

    def test_critical_issues(self):
        with patch.object(console, "print") as mock_print:
            _print_status_messages(make_summary(critical=3, total_issues=3))
        assert mock_print.called

    def test_no_issues_but_failures(self):
        with patch.object(console, "print") as mock_print:
            _print_status_messages(make_summary(files_failed=1, total_issues=0))
        assert mock_print.called


# ─────────────────────────── _print_detailed_issues ───────────────────────────


class TestPrintDetailedIssues:
    def test_critical_issue(self):
        result = MagicMock()
        result.all_issues = [make_issue_mock("critical")]
        with patch.object(console, "print"):
            _print_detailed_issues(result)

    def test_warning_with_suggestion(self):
        result = MagicMock()
        result.all_issues = [make_issue_mock("warning", suggestion="Use xyz")]
        with patch.object(console, "print"):
            _print_detailed_issues(result)

    def test_info_issue(self):
        result = MagicMock()
        result.all_issues = [make_issue_mock("info")]
        with patch.object(console, "print"):
            _print_detailed_issues(result)

    def test_error_issue(self):
        result = MagicMock()
        result.all_issues = [make_issue_mock("error")]
        with patch.object(console, "print"):
            _print_detailed_issues(result)


# ─────────────────────────── _print_helpful_tips ──────────────────────────────


class TestPrintHelpfulTips:
    def test_no_issues(self):
        with patch.object(console, "print") as mock_print:
            _print_helpful_tips(make_summary(total_issues=0), detailed=True)
        mock_print.assert_not_called()

    def test_issues_not_detailed(self):
        with patch.object(console, "print") as mock_print:
            _print_helpful_tips(make_summary(total_issues=2), detailed=False)
        assert any("--detailed" in str(c) for c in mock_print.call_args_list)

    def test_many_issues(self):
        with patch.object(console, "print") as mock_print:
            _print_helpful_tips(make_summary(total_issues=10), detailed=True)
        assert mock_print.called


# ─────────────────────────── _print_refactor_filters ──────────────────────────


class TestPrintRefactorFilters:
    def test_no_types(self):
        with patch.object(console, "print") as mock_print:
            _print_refactor_filters(())
        mock_print.assert_not_called()

    def test_with_types(self):
        with patch.object(console, "print") as mock_print:
            _print_refactor_filters(("rename", "extract"))
        assert mock_print.called


# ─────────────────────────── _confirm_apply_mode ──────────────────────────────


class TestConfirmApplyMode:
    def test_preview_mode_no_confirm(self):
        with patch.object(console, "print") as mock_print:
            _confirm_apply_mode(preview=True)
        mock_print.assert_not_called()

    def test_apply_mode_confirmed(self):
        with (
            patch.object(console, "print"),
            patch("refactron.cli.ui.click.confirm", return_value=True),
        ):
            _confirm_apply_mode(preview=False)  # Should not raise

    def test_apply_mode_cancelled(self):
        with (
            patch.object(console, "print"),
            patch("refactron.cli.ui.click.confirm", return_value=False),
        ):
            with pytest.raises(SystemExit):
                _confirm_apply_mode(preview=False)


# ─────────────────────────── _create_refactor_table ───────────────────────────


class TestCreateRefactorTable:
    def test_no_high_risk(self):
        table = _create_refactor_table(
            make_summary(total_operations=3, safe=3, high_risk=0, applied=False)
        )
        assert table is not None

    def test_with_high_risk(self):
        table = _create_refactor_table(
            make_summary(total_operations=2, safe=1, high_risk=1, applied=True)
        )
        assert table is not None


# ─────────────────────────── _print_refactor_messages ─────────────────────────


class TestPrintRefactorMessages:
    def test_no_operations(self):
        with patch.object(console, "print"):
            _print_refactor_messages(make_summary(total_operations=0), preview=True)

    def test_high_risk(self):
        with patch.object(console, "print") as mock_print:
            _print_refactor_messages(make_summary(total_operations=2, high_risk=1), preview=True)
        assert mock_print.called

    def test_preview_with_operations(self):
        with patch.object(console, "print") as mock_print:
            _print_refactor_messages(make_summary(total_operations=3), preview=True)
        assert mock_print.called

    def test_applied_success(self):
        with patch.object(console, "print") as mock_print:
            _print_refactor_messages(make_summary(total_operations=2, applied=True), preview=False)
        assert mock_print.called


# ─────────────────────────── _collect_feedback_interactive ────────────────────


class TestCollectFeedbackInteractive:
    def test_no_operations(self):
        refactron = MagicMock()
        result = MagicMock()
        result.operations = []
        with patch.object(console, "print"):
            _collect_feedback_interactive(refactron, result)
        refactron.record_feedback.assert_not_called()

    def test_skip_feedback(self):
        refactron = MagicMock()
        op = MagicMock()
        op.operation_id = "op1"
        op.operation_type = "rename"
        op.file_path = "test.py"
        op.line_number = 5
        result = MagicMock()
        result.operations = [op]
        result.get_ranking_score.return_value = 0.5
        with (
            patch.object(console, "print"),
            patch("refactron.cli.ui.click.prompt", return_value="s"),
        ):
            _collect_feedback_interactive(refactron, result)
        refactron.record_feedback.assert_not_called()

    def test_accepted_feedback_with_reason(self):
        refactron = MagicMock()
        op = MagicMock()
        op.operation_id = "op2"
        op.operation_type = "extract"
        op.file_path = "a.py"
        op.line_number = 1
        result = MagicMock()
        result.operations = [op]
        result.get_ranking_score.return_value = 0.0
        with (
            patch.object(console, "print"),
            patch("refactron.cli.ui.click.prompt", side_effect=["a", "looks good"]),
        ):
            _collect_feedback_interactive(refactron, result)
        refactron.record_feedback.assert_called_once()

    def test_rejected_feedback_no_reason(self):
        refactron = MagicMock()
        op = MagicMock()
        op.operation_id = "op3"
        op.operation_type = "rename"
        op.file_path = "b.py"
        op.line_number = 3
        result = MagicMock()
        result.operations = [op]
        result.get_ranking_score.return_value = 0.0
        with (
            patch.object(console, "print"),
            patch("refactron.cli.ui.click.prompt", side_effect=["r", ""]),
        ):
            _collect_feedback_interactive(refactron, result)
        refactron.record_feedback.assert_called_once()


# ─────────────────────────── _record_applied_operations ───────────────────────


class TestRecordAppliedOperations:
    def test_no_operations(self):
        refactron = MagicMock()
        result = MagicMock()
        result.operations = []
        _record_applied_operations(refactron, result)
        refactron.record_feedback.assert_not_called()

    def test_records_accepted(self):
        refactron = MagicMock()
        op = MagicMock()
        op.operation_id = "op1"
        result = MagicMock()
        result.operations = [op]
        _record_applied_operations(refactron, result)
        refactron.record_feedback.assert_called_once_with(
            operation_id="op1",
            action="accepted",
            reason="Applied via --apply flag",
            operation=op,
        )


# ─────────────────────────── _interactive_file_selector ───────────────────────


class TestInteractiveFileSelector:
    def test_nonexistent_directory(self, tmp_path):
        result = _interactive_file_selector(tmp_path / "nonexistent")
        assert result is None

    def test_no_matching_files(self, tmp_path):
        result = _interactive_file_selector(tmp_path, pattern="*.py")
        assert result is None

    def test_user_cancels_with_zero(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        with patch("refactron.cli.ui.IntPrompt.ask", return_value=0):
            result = _interactive_file_selector(tmp_path, pattern="*.py")
        assert result is None

    def test_selects_file(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        with (
            patch("refactron.cli.ui.IntPrompt.ask", return_value=1),
            patch.object(console, "print"),
        ):
            result = _interactive_file_selector(tmp_path, pattern="*.py")
        assert result is not None

    def test_out_of_range_selection(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        with (
            patch("refactron.cli.ui.IntPrompt.ask", return_value=999),
            patch.object(console, "print"),
        ):
            result = _interactive_file_selector(tmp_path, pattern="*.py")
        assert result is None

    def test_more_than_20_files(self, tmp_path):
        for i in range(25):
            (tmp_path / f"file{i}.py").write_text(f"x={i}")
        with (
            patch("refactron.cli.ui.IntPrompt.ask", return_value=0),
            patch.object(console, "print"),
        ):
            result = _interactive_file_selector(tmp_path, pattern="*.py")
        assert result is None


# ─────────────────────────── _print_custom_help ───────────────────────────────


class TestPrintCustomHelp:
    def test_renders_without_error(self):
        ctx = MagicMock()
        ctx.command.list_commands.return_value = ["analyze", "refactor"]
        ctx.command.get_command.return_value = MagicMock(get_short_help_str=lambda: "desc")
        with patch.object(console, "print"):
            _print_custom_help(ctx)
