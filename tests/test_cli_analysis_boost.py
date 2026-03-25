"""
Tests for cli/analysis.py - covers analyze, report, metrics, serve_metrics, suggest commands.
All LLM/network/file-system I/O is mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from refactron.cli.analysis import analyze, metrics, report, serve_metrics, suggest


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def mock_cfg():
    cfg = MagicMock()
    cfg.log_level = "INFO"
    cfg.log_format = "text"
    cfg.enable_metrics = False
    cfg.backup_enabled = True
    cfg.report_format = "text"
    return cfg


@pytest.fixture()
def mock_analysis_result():
    result = MagicMock()
    result.all_issues = []
    result.summary.return_value = {
        "total_files": 1,
        "files_analyzed": 1,
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
    result.report.return_value = "# Report\n"
    return result


# ─────────────────────────── analyze command ──────────────────────────────────


class TestAnalyzeCommand:
    def test_analyze_no_target_no_workspace(self, runner, tmp_path, mock_cfg):
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis.WorkspaceManager") as mock_ws,
        ):
            mock_ws.return_value.get_workspace_by_path.return_value = None
            result = runner.invoke(analyze, [])
        assert result.exit_code == 1

    def test_analyze_with_target(self, runner, tmp_path, mock_cfg, mock_analysis_result):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis._validate_path", return_value=tmp_path),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._print_file_count"),
            patch("refactron.cli.analysis._create_summary_table", return_value=MagicMock()),
            patch("refactron.cli.analysis._print_status_messages"),
            patch("refactron.cli.analysis._print_helpful_tips"),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.analyze.return_value = mock_analysis_result
            result = runner.invoke(analyze, [str(tmp_path)])
        assert result.exit_code == 0

    def test_analyze_with_critical_issues(self, runner, tmp_path, mock_cfg, mock_analysis_result):
        mock_analysis_result.summary.return_value["critical"] = 2
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis._validate_path", return_value=tmp_path),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._print_file_count"),
            patch("refactron.cli.analysis._create_summary_table", return_value=MagicMock()),
            patch("refactron.cli.analysis._print_status_messages"),
            patch("refactron.cli.analysis._print_helpful_tips"),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.analyze.return_value = mock_analysis_result
            result = runner.invoke(analyze, [str(tmp_path)])
        assert result.exit_code == 1

    def test_analyze_exception_raises_systemexit(self, runner, tmp_path, mock_cfg):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis._validate_path", return_value=tmp_path),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._print_file_count"),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.analyze.side_effect = RuntimeError("fail")
            result = runner.invoke(analyze, [str(tmp_path)])
        assert result.exit_code == 1

    def test_analyze_show_metrics(self, runner, tmp_path, mock_cfg, mock_analysis_result):
        mock_cfg.enable_metrics = True
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        mock_collector = MagicMock()
        mock_collector.get_analysis_summary.return_value = {
            "total_analysis_time_ms": 100.0,
            "average_time_per_file_ms": 50.0,
            "success_rate_percent": 100.0,
        }
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis._validate_path", return_value=tmp_path),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._print_file_count"),
            patch("refactron.cli.analysis._create_summary_table", return_value=MagicMock()),
            patch("refactron.cli.analysis._print_status_messages"),
            patch("refactron.cli.analysis._print_helpful_tips"),
            patch("refactron.cli.analysis._auth_banner"),
            # Lazy-imported inside analyze; patch at the source module
            patch("refactron.core.metrics.get_metrics_collector", return_value=mock_collector),
        ):
            mock_ref.return_value.analyze.return_value = mock_analysis_result
            result = runner.invoke(analyze, [str(tmp_path), "--show-metrics"])
        assert result.exit_code == 0

    def test_analyze_workspace_connected(self, runner, tmp_path, mock_cfg, mock_analysis_result):
        mock_workspace = MagicMock()
        mock_workspace.repo_full_name = "user/repo"
        mock_workspace.local_path = str(tmp_path)
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis.WorkspaceManager") as mock_ws,
            patch(
                "refactron.cli.analysis._interactive_file_selector", return_value=tmp_path / "f.py"
            ),
            patch("refactron.cli.analysis._validate_path", return_value=tmp_path),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._print_file_count"),
            patch("refactron.cli.analysis._create_summary_table", return_value=MagicMock()),
            patch("refactron.cli.analysis._print_status_messages"),
            patch("refactron.cli.analysis._print_helpful_tips"),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ws.return_value.get_workspace_by_path.return_value = mock_workspace
            mock_ref.return_value.analyze.return_value = mock_analysis_result
            result = runner.invoke(analyze, [])
        assert result.exit_code == 0


# ─────────────────────────── report command ───────────────────────────────────


class TestReportCommand:
    def test_report_text_to_stdout(self, runner, tmp_path, mock_cfg, mock_analysis_result):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.analyze.return_value = mock_analysis_result
            result = runner.invoke(report, [str(tmp_path)])
        assert result.exit_code == 0

    def test_report_to_file(self, runner, tmp_path, mock_cfg, mock_analysis_result):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        output_file = tmp_path / "report.txt"
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.analyze.return_value = mock_analysis_result
            result = runner.invoke(report, [str(tmp_path), "--output", str(output_file)])
        assert result.exit_code == 0

    def test_report_failure(self, runner, tmp_path, mock_cfg):
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n")
        with (
            patch("refactron.cli.analysis._load_config", return_value=mock_cfg),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.analyze.side_effect = RuntimeError("oops")
            result = runner.invoke(report, [str(tmp_path)])
        assert result.exit_code == 1


# ─────────────────────────── metrics command ──────────────────────────────────


class TestMetricsCommand:
    def test_metrics_text_format(self, runner):
        mock_collector = MagicMock()
        mock_collector.get_analysis_summary.return_value = {
            "total_files_analyzed": 5,
            "total_files_failed": 0,
            "total_issues_found": 2,
            "total_analysis_time_ms": 100.0,
            "average_time_per_file_ms": 20.0,
            "success_rate_percent": 100.0,
            "analyzer_hit_counts": {"complexity": 3},
        }
        mock_collector.get_combined_summary.return_value = {
            "analysis": mock_collector.get_analysis_summary.return_value,
            "refactoring": {"total_refactorings_applied": 0, "success_rate_percent": 100.0},
        }
        # metrics is lazy-imported inside the command; patch at source
        with (
            patch("refactron.core.metrics.get_metrics_collector", return_value=mock_collector),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            result = runner.invoke(metrics, [])
        assert result.exit_code == 0

    def test_metrics_json_format(self, runner):
        mock_collector = MagicMock()
        mock_collector.get_combined_summary.return_value = {"analysis": {}, "refactoring": {}}
        with (
            patch("refactron.core.metrics.get_metrics_collector", return_value=mock_collector),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            result = runner.invoke(metrics, ["--format", "json"])
        assert result.exit_code == 0


# ─────────────────────────── serve_metrics command ────────────────────────────


class TestServeMetricsCommand:
    def test_serve_metrics_keyboard_interrupt(self, runner):
        # Patch at the lazy-import source modules
        with (
            patch("refactron.core.prometheus_metrics.start_metrics_server"),
            patch("refactron.core.prometheus_metrics.stop_metrics_server"),
            patch("refactron.cli.analysis._auth_banner"),
            patch("time.sleep", side_effect=KeyboardInterrupt),
        ):
            result = runner.invoke(serve_metrics, [])
        assert result.exit_code == 0

    def test_serve_metrics_start_failure(self, runner):
        with (
            patch(
                "refactron.core.prometheus_metrics.start_metrics_server",
                side_effect=RuntimeError("port busy"),
            ),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            result = runner.invoke(serve_metrics, [])
        assert result.exit_code == 1


# ─────────────────────────── suggest command ──────────────────────────────────


class TestSuggestCommand:
    def test_suggest_on_file(self, runner, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("def foo():\n    x = 1\n    return x\n")

        from refactron.llm.models import SuggestionStatus

        mock_sugg = MagicMock()
        mock_sugg.status = SuggestionStatus.PENDING
        mock_sugg.explanation = "Use constant"
        mock_sugg.proposed_code = "def foo():\n    return 1\n"
        mock_sugg.model_name = "test-model"
        mock_sugg.confidence_score = 0.9
        mock_sugg.llm_confidence = 0.9
        mock_sugg.safety_result = None

        with (
            patch("refactron.cli.analysis._load_config"),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis.ContextRetriever"),
            patch("refactron.cli.analysis.LLMOrchestrator") as mock_orch,
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            mock_orch.return_value.generate_suggestion.return_value = mock_sugg
            result = runner.invoke(suggest, [str(py_file)])
        assert result.exit_code == 0

    def test_suggest_directory_not_supported(self, runner, tmp_path):
        with (
            patch("refactron.cli.analysis._load_config"),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis.ContextRetriever"),
            patch("refactron.cli.analysis.LLMOrchestrator"),
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            result = runner.invoke(suggest, [str(tmp_path)])
        assert result.exit_code == 0

    def test_suggest_failed_status(self, runner, tmp_path):
        py_file = tmp_path / "bad.py"
        py_file.write_text("x = 1\n")

        from refactron.llm.models import SuggestionStatus

        mock_sugg = MagicMock()
        mock_sugg.status = SuggestionStatus.FAILED
        mock_sugg.explanation = "No LLM"

        with (
            patch("refactron.cli.analysis._load_config"),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis.ContextRetriever"),
            patch("refactron.cli.analysis.LLMOrchestrator") as mock_orch,
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            mock_orch.return_value.generate_suggestion.return_value = mock_sugg
            result = runner.invoke(suggest, [str(py_file)])
        assert result.exit_code == 0

    def test_suggest_with_line(self, runner, tmp_path):
        py_file = tmp_path / "code.py"
        py_file.write_text("\n".join(["x = 0"] * 20) + "\n")

        from refactron.llm.models import SuggestionStatus

        mock_sugg = MagicMock()
        mock_sugg.status = SuggestionStatus.PENDING
        mock_sugg.explanation = "Better"
        mock_sugg.proposed_code = "x = 0\n"
        mock_sugg.model_name = "model"
        mock_sugg.confidence_score = 0.9
        mock_sugg.llm_confidence = 0.9
        mock_sugg.safety_result = None

        with (
            patch("refactron.cli.analysis._load_config"),
            patch("refactron.cli.analysis._setup_logging"),
            patch("refactron.cli.analysis.Refactron") as mock_ref,
            patch("refactron.cli.analysis.ContextRetriever"),
            patch("refactron.cli.analysis.LLMOrchestrator") as mock_orch,
            patch("refactron.cli.analysis._auth_banner"),
        ):
            mock_ref.return_value.detect_project_root.return_value = tmp_path
            mock_orch.return_value.generate_suggestion.return_value = mock_sugg
            result = runner.invoke(suggest, [str(py_file), "--line", "5"])
        assert result.exit_code == 0
