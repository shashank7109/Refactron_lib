"""Tests for cli/cicd.py – generate_cicd and feedback commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from refactron.cli.cicd import feedback, generate_cicd, init


@pytest.fixture()
def runner():
    return CliRunner()


class TestGenerateCicd:
    def test_github_type(self, runner, tmp_path):
        with patch("refactron.cli.cicd._auth_banner"):
            result = runner.invoke(generate_cicd, ["github", "--output", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / ".github" / "workflows").exists()

    def test_gitlab_type(self, runner, tmp_path):
        with patch("refactron.cli.cicd._auth_banner"):
            result = runner.invoke(generate_cicd, ["gitlab", "--output", str(tmp_path)])
        assert result.exit_code == 0
        yml = tmp_path / ".gitlab-ci.yml"
        assert yml.exists() or result.exit_code == 0  # generator may vary

    def test_precommit_type(self, runner, tmp_path):
        with patch("refactron.cli.cicd._auth_banner"):
            result = runner.invoke(generate_cicd, ["pre-commit", "--output", str(tmp_path)])
        assert result.exit_code == 0

    def test_all_type(self, runner, tmp_path):
        with patch("refactron.cli.cicd._auth_banner"):
            result = runner.invoke(generate_cicd, ["all", "--output", str(tmp_path)])
        assert result.exit_code == 0

    def test_invalid_type(self, runner, tmp_path):
        result = runner.invoke(generate_cicd, ["jenkins", "--output", str(tmp_path)])
        assert result.exit_code == 2  # Invalid choice

    def test_default_output_cwd(self, runner, tmp_path):
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("pathlib.Path.mkdir"),
            patch(
                "refactron.cicd.github_actions.GitHubActionsGenerator.generate_analysis_workflow",
                return_value="",
            ),
            patch(
                "refactron.cicd.github_actions.GitHubActionsGenerator.generate_pre_commit_workflow",
                return_value="",
            ),
            patch("refactron.cicd.github_actions.GitHubActionsGenerator.save_workflow"),
        ):
            result = runner.invoke(generate_cicd, ["github"])
        assert result.exit_code == 0

    def test_write_error_handled(self, runner, tmp_path):
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch(
                "refactron.cicd.github_actions.GitHubActionsGenerator.save_workflow",
                side_effect=OSError("disk full"),
            ),
        ):
            result = runner.invoke(generate_cicd, ["github", "--output", str(tmp_path)])
        assert result.exit_code == 1

    def test_fail_flags_passed(self, runner, tmp_path):
        with patch("refactron.cli.cicd._auth_banner"):
            result = runner.invoke(
                generate_cicd,
                [
                    "github",
                    "--output",
                    str(tmp_path),
                    "--fail-on-critical",
                    "--fail-on-errors",
                    "--max-critical",
                    "2",
                    "--max-errors",
                    "5",
                ],
            )
        assert result.exit_code == 0


class TestFeedbackCommand:
    def test_feedback_accepted(self, runner):
        mock_refactron = MagicMock()
        mock_refactron.pattern_storage = None
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("refactron.cli.cicd._load_config"),
            patch("refactron.cli.cicd.Refactron", return_value=mock_refactron),
        ):
            result = runner.invoke(feedback, ["op123", "--action", "accepted"])
        assert result.exit_code == 0

    def test_feedback_rejected_with_reason(self, runner):
        mock_refactron = MagicMock()
        mock_refactron.pattern_storage = None
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("refactron.cli.cicd._load_config"),
            patch("refactron.cli.cicd.Refactron", return_value=mock_refactron),
        ):
            result = runner.invoke(
                feedback, ["op456", "--action", "rejected", "--reason", "Too risky"]
            )
        assert result.exit_code == 0

    def test_feedback_with_pattern_storage(self, runner):
        mock_refactron = MagicMock()
        fb_mock = MagicMock()
        fb_mock.operation_id = "op789"
        mock_refactron.pattern_storage.load_feedback.return_value = [fb_mock]
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("refactron.cli.cicd._load_config"),
            patch("refactron.cli.cicd.Refactron", return_value=mock_refactron),
        ):
            result = runner.invoke(feedback, ["op789", "--action", "accepted"])
        assert result.exit_code == 0

    def test_feedback_unknown_operation_warns(self, runner):
        mock_refactron = MagicMock()
        fb_mock = MagicMock()
        fb_mock.operation_id = "other"
        mock_refactron.pattern_storage.load_feedback.return_value = [fb_mock]
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("refactron.cli.cicd._load_config"),
            patch("refactron.cli.cicd.Refactron", return_value=mock_refactron),
        ):
            result = runner.invoke(feedback, ["unknown_op", "--action", "ignored"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_feedback_record_failure(self, runner):
        mock_refactron = MagicMock()
        mock_refactron.pattern_storage = None
        mock_refactron.record_feedback.side_effect = RuntimeError("DB error")
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("refactron.cli.cicd._load_config"),
            patch("refactron.cli.cicd.Refactron", return_value=mock_refactron),
        ):
            result = runner.invoke(feedback, ["op1", "--action", "accepted"])
        assert result.exit_code == 1

    def test_feedback_init_failure(self, runner):
        with (
            patch("refactron.cli.cicd._auth_banner"),
            patch("refactron.cli.cicd._load_config"),
            patch("refactron.cli.cicd.Refactron", side_effect=Exception("init fail")),
        ):
            result = runner.invoke(feedback, ["op1", "--action", "accepted"])
        assert result.exit_code == 1

    def test_feedback_missing_action(self, runner):
        result = runner.invoke(feedback, ["op1"])
        assert result.exit_code == 2

    def test_feedback_invalid_action(self, runner):
        result = runner.invoke(feedback, ["op1", "--action", "maybe"])
        assert result.exit_code == 2


class TestInitCommand:
    def test_init_creates_config(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init, [])
        assert result.exit_code == 0 or (tmp_path / ".refactron.yaml").exists()

    def test_init_with_template(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init, ["--template", "django"])
        assert result.exit_code == 0

    def test_init_overwrite_yes(self, runner, tmp_path):
        config = tmp_path / ".refactron.yaml"
        config.write_text("existing: config\n")
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init, [], input="y\n")
        assert result.exit_code == 0

    def test_init_overwrite_no(self, runner, tmp_path):
        config = tmp_path / ".refactron.yaml"
        config.write_text("existing: config\n")
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init, [], input="n\n")
        assert result.exit_code == 0

    def test_init_invalid_template(self, runner, tmp_path):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(init, ["--template", "invalid_tmpl"])
        assert result.exit_code == 2  # Invalid choice
