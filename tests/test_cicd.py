"""Tests for CI/CD integration templates and utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from refactron.cicd.github_actions import GitHubActionsGenerator
from refactron.cicd.gitlab_ci import GitLabCIGenerator
from refactron.cicd.pr_integration import PRComment, PRIntegration
from refactron.cicd.pre_commit import PreCommitGenerator
from refactron.cicd.quality_gates import QualityGate, QualityGateParser
from refactron.core.analysis_result import AnalysisResult
from refactron.core.models import CodeIssue, FileMetrics, IssueCategory, IssueLevel


class TestQualityGate:
    """Tests for QualityGate class."""

    def test_quality_gate_defaults(self) -> None:
        """Test quality gate default values."""
        gate = QualityGate()
        assert gate.max_critical == 0
        assert gate.max_errors == 10
        assert gate.max_warnings == 50
        assert gate.fail_on_critical is True
        assert gate.fail_on_errors is False
        assert gate.min_success_rate == 0.95

    def test_quality_gate_check_passes(self) -> None:
        """Test quality gate check with passing result."""
        gate = QualityGate(max_critical=0, fail_on_critical=True)

        # Create a result with no issues
        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=Path("test.py"),
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=[],
                )
            ],
            total_files=1,
            total_issues=0,
        )

        passed, message = gate.check(result)
        assert passed is True
        assert "passed" in message.lower()

    def test_quality_gate_check_fails_critical(self) -> None:
        """Test quality gate check fails on critical issues."""
        gate = QualityGate(max_critical=0, fail_on_critical=True)

        # Create a result with critical issues
        critical_issue = CodeIssue(
            file_path=Path("test.py"),
            line_number=1,
            message="Critical issue",
            rule_id="TEST001",
            level=IssueLevel.CRITICAL,
            category=IssueCategory.SECURITY,
        )

        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=Path("test.py"),
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=[critical_issue],
                )
            ],
            total_files=1,
            total_issues=1,
        )

        passed, message = gate.check(result)
        assert passed is False
        assert "critical" in message.lower()

    def test_quality_gate_check_fails_threshold(self) -> None:
        """Test quality gate check fails when threshold exceeded."""
        gate = QualityGate(max_critical=1, max_errors=5)

        # Create a result with too many critical issues
        issues = [
            CodeIssue(
                file_path=Path("test.py"),
                line_number=i,
                message=f"Issue {i}",
                rule_id="TEST001",
                level=IssueLevel.CRITICAL,
                category=IssueCategory.SECURITY,
            )
            for i in range(1, 4)  # 3 critical issues > 1 threshold
        ]

        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=Path("test.py"),
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=issues,
                )
            ],
            total_files=1,
            total_issues=3,
        )

        passed, message = gate.check(result)
        assert passed is False
        assert "exceed" in message.lower()

    def test_quality_gate_to_dict(self) -> None:
        """Test quality gate conversion to dictionary."""
        gate = QualityGate(max_critical=1, max_errors=5, fail_on_critical=True)
        gate_dict = gate.to_dict()

        assert gate_dict["max_critical"] == 1
        assert gate_dict["max_errors"] == 5
        assert gate_dict["fail_on_critical"] is True

    def test_quality_gate_from_dict(self) -> None:
        """Test quality gate creation from dictionary."""
        gate_dict = {
            "max_critical": 2,
            "max_errors": 10,
            "fail_on_critical": False,
        }
        gate = QualityGate.from_dict(gate_dict)

        assert gate.max_critical == 2
        assert gate.max_errors == 10
        assert gate.fail_on_critical is False


class TestQualityGateParser:
    """Tests for QualityGateParser class."""

    def test_parse_json_output(self) -> None:
        """Test parsing JSON output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "summary": {
                    "critical": 2,
                    "errors": 5,
                    "warnings": 10,
                    "total_issues": 17,
                }
            }
            json.dump(data, f)
            json_path = Path(f.name)

        try:
            result = QualityGateParser.parse_json_output(json_path)
            assert result["summary"]["critical"] == 2
            assert result["summary"]["errors"] == 5
        finally:
            json_path.unlink()

    def test_parse_json_output_not_found(self) -> None:
        """Test parsing JSON output when file doesn't exist."""
        json_path = Path("/nonexistent/file.json")

        with pytest.raises(FileNotFoundError):
            QualityGateParser.parse_json_output(json_path)

    def test_parse_text_output(self) -> None:
        """Test parsing text output."""
        text = """
        Critical: 5
        Errors: 10
        Warnings: 20
        Total: 35
        """

        summary = QualityGateParser.parse_text_output(text)
        assert summary["critical"] == 5
        assert summary["errors"] == 10
        assert summary["warnings"] == 20

    def test_parse_exit_code(self) -> None:
        """Test parsing exit code."""
        result = QualityGateParser.parse_exit_code(0)
        assert result["exit_code"] == 0
        assert result["should_fail"] is False

        result = QualityGateParser.parse_exit_code(1)
        assert result["exit_code"] == 1
        assert result["should_fail"] is True

    def test_enforce_gate_passes(self) -> None:
        """Test enforcing quality gate with passing result."""
        gate = QualityGate(max_critical=0)

        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=Path("test.py"),
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=[],
                )
            ],
            total_files=1,
            total_issues=0,
        )

        passed, message, exit_code = QualityGateParser.enforce_gate(result, gate)
        assert passed is True
        assert exit_code == 0

    def test_enforce_gate_fails(self) -> None:
        """Test enforcing quality gate with failing result."""
        gate = QualityGate(max_critical=0, fail_on_critical=True)

        critical_issue = CodeIssue(
            file_path=Path("test.py"),
            line_number=1,
            message="Critical",
            rule_id="TEST001",
            level=IssueLevel.CRITICAL,
            category=IssueCategory.SECURITY,
        )

        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=Path("test.py"),
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=[critical_issue],
                )
            ],
            total_files=1,
            total_issues=1,
        )

        passed, message, exit_code = QualityGateParser.enforce_gate(result, gate)
        assert passed is False
        assert exit_code == 1


class TestGitHubActionsGenerator:
    """Tests for GitHubActionsGenerator class."""

    def test_generate_analysis_workflow(self) -> None:
        """Test generating GitHub Actions analysis workflow."""
        generator = GitHubActionsGenerator()
        workflow = generator.generate_analysis_workflow(python_versions=["3.10", "3.11"])

        assert "name: Refactron Code Analysis" in workflow
        assert "3.10" in workflow
        assert "3.11" in workflow
        assert "refactron analyze" in workflow
        assert "quality gate" in workflow.lower()

    def test_generate_pre_commit_workflow(self) -> None:
        """Test generating GitHub Actions pre-commit workflow."""
        generator = GitHubActionsGenerator()
        workflow = generator.generate_pre_commit_workflow()

        assert "name: Refactron Pre-Commit" in workflow
        assert "refactron analyze" in workflow

    def test_save_workflow(self) -> None:
        """Test saving workflow to file."""
        generator = GitHubActionsGenerator()
        workflow_content = generator.generate_analysis_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "workflow.yml"
            generator.save_workflow(workflow_content, output_path)

            assert output_path.exists()
            assert "Refactron Code Analysis" in output_path.read_text(encoding="utf-8")


class TestGitLabCIGenerator:
    """Tests for GitLabCIGenerator class."""

    def test_generate_analysis_pipeline(self) -> None:
        """Test generating GitLab CI analysis pipeline."""
        generator = GitLabCIGenerator()
        pipeline = generator.generate_analysis_pipeline(python_versions=["3.10", "3.11"])

        assert "stages:" in pipeline
        assert "analyze:" in pipeline
        assert "refactron analyze" in pipeline
        assert "python:3.10" in pipeline

    def test_generate_pre_commit_pipeline(self) -> None:
        """Test generating GitLab CI pre-commit pipeline."""
        generator = GitLabCIGenerator()
        pipeline = generator.generate_pre_commit_pipeline()

        assert "stages:" in pipeline
        assert "pre-commit:" in pipeline
        assert "refactron analyze" in pipeline

    def test_save_pipeline(self) -> None:
        """Test saving pipeline to file."""
        generator = GitLabCIGenerator()
        pipeline_content = generator.generate_analysis_pipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / ".gitlab-ci.yml"
            generator.save_pipeline(pipeline_content, output_path)

            assert output_path.exists()
            assert "analyze:" in output_path.read_text(encoding="utf-8")


class TestPreCommitGenerator:
    """Tests for PreCommitGenerator class."""

    def test_generate_pre_commit_config(self) -> None:
        """Test generating pre-commit configuration."""
        generator = PreCommitGenerator()
        config = generator.generate_pre_commit_config()

        assert "repos:" in config
        assert "refactron-analyze" in config
        assert "refactron-quality-gate" in config

    def test_generate_simple_hook(self) -> None:
        """Test generating simple pre-commit hook."""
        generator = PreCommitGenerator()
        hook = generator.generate_simple_hook()

        assert "#!/bin/bash" in hook
        assert "refactron analyze" in hook

    def test_save_config(self) -> None:
        """Test saving pre-commit configuration."""
        generator = PreCommitGenerator()
        config_content = generator.generate_pre_commit_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / ".pre-commit-config.yaml"
            generator.save_config(config_content, output_path)

            assert output_path.exists()
            assert "refactron" in output_path.read_text(encoding="utf-8")

    def test_save_hook(self) -> None:
        """Test saving pre-commit hook script."""
        import sys

        generator = PreCommitGenerator()
        hook_content = generator.generate_simple_hook()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pre-commit"
            generator.save_hook(hook_content, output_path)

            assert output_path.exists()
            # chmod(0o755) has no effect on Windows; skip the bit-check there.
            if sys.platform != "win32":
                assert output_path.stat().st_mode & 0o111  # Executable


class TestPRIntegration:
    """Tests for PRIntegration class."""

    def test_generate_pr_summary(self) -> None:
        """Test generating PR summary."""
        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=Path("test.py"),
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=[
                        CodeIssue(
                            file_path=Path("test.py"),
                            line_number=1,
                            message="Issue",
                            rule_id="TEST001",
                            level=IssueLevel.WARNING,
                            category=IssueCategory.CODE_SMELL,
                        )
                    ],
                )
            ],
            total_files=1,
            total_issues=1,
        )

        summary = PRIntegration.generate_pr_summary(result)
        assert "Refactron Analysis Summary" in summary
        assert "test.py" in summary

    def test_generate_inline_comments(self) -> None:
        """Test generating inline comments."""
        file_path = Path("test.py")
        issue = CodeIssue(
            file_path=file_path,
            line_number=10,
            message="Test issue",
            rule_id="TEST001",
            level=IssueLevel.ERROR,
            category=IssueCategory.SECURITY,
            suggestion="Fix this",
        )

        result = AnalysisResult(
            file_metrics=[
                FileMetrics(
                    file_path=file_path,
                    lines_of_code=10,
                    comment_lines=2,
                    blank_lines=1,
                    complexity=1.0,
                    maintainability_index=100.0,
                    functions=1,
                    classes=0,
                    issues=[issue],
                )
            ],
            total_files=1,
            total_issues=1,
        )

        comments = PRIntegration.generate_inline_comments(result, file_path)
        assert len(comments) == 1
        assert comments[0].file_path == str(file_path)
        assert comments[0].line == 10
        assert comments[0].message == "Test issue"

    def test_format_comment_for_github_api(self) -> None:
        """Test formatting comment for GitHub API."""
        comment = PRComment(
            file_path="test.py",
            line=5,
            message="Test message",
            level="error",
            rule_id="TEST001",
            suggestion="Fix: use better approach",
        )

        formatted = PRIntegration.format_comment_for_github_api(comment)
        assert formatted["path"] == "test.py"
        assert formatted["line"] == 5
        assert "Test message" in formatted["body"]
        assert "Fix: use better approach" in formatted["body"]
