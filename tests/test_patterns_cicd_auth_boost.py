"""Tests for patterns/learner.py, cicd/quality_gates.py, cicd/pr_integration.py, cli/auth.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# ──────────────────────────── PatternLearner ──────────────────────────────────


class TestPatternLearnerInit:
    def test_raises_on_none_storage(self):
        from refactron.patterns.learner import PatternLearner

        with pytest.raises(ValueError, match="PatternStorage cannot be None"):
            PatternLearner(storage=None, fingerprinter=MagicMock())

    def test_raises_on_none_fingerprinter(self):
        from refactron.patterns.learner import PatternLearner

        with pytest.raises(ValueError, match="PatternFingerprinter cannot be None"):
            PatternLearner(storage=MagicMock(), fingerprinter=None)


def make_learner():
    from refactron.patterns.learner import PatternLearner

    storage = MagicMock()
    storage.load_patterns.return_value = {}
    storage.get_pattern_metric.return_value = None
    fingerprinter = MagicMock()
    fingerprinter.fingerprint_code.return_value = "hash123"
    return PatternLearner(storage=storage, fingerprinter=fingerprinter), storage, fingerprinter


def make_operation(has_hash=True):
    op = MagicMock()
    op.operation_id = "op1"
    op.operation_type = "rename"
    op.old_code = "def foo(): pass"
    op.new_code = "def bar(): pass"
    op.risk_score = 0.2
    op.metadata = {"code_pattern_hash": "h1"} if has_hash else {}
    return op


def make_feedback(action="accepted"):
    fb = MagicMock()
    fb.action = action
    fb.project_path = None
    return fb


class TestLearnFromFeedback:
    def test_none_operation_raises(self):
        learner, _, _ = make_learner()
        with pytest.raises(ValueError):
            learner.learn_from_feedback(None, make_feedback())

    def test_none_feedback_raises(self):
        learner, _, _ = make_learner()
        with pytest.raises(ValueError):
            learner.learn_from_feedback(make_operation(), None)

    def test_creates_new_pattern(self):
        learner, storage, _ = make_learner()
        storage.load_patterns.return_value = {}
        mock_pattern = MagicMock()
        mock_pattern.pattern_id = "p1"
        with patch("refactron.patterns.learner.RefactoringPattern") as mock_cls:
            mock_cls.create.return_value = mock_pattern
            result = learner.learn_from_feedback(make_operation(), make_feedback())
        assert result == "p1"

    def test_updates_existing_pattern(self):
        learner, storage, _ = make_learner()
        existing_pattern = MagicMock()
        existing_pattern.pattern_hash = "h1"
        existing_pattern.operation_type = "rename"
        existing_pattern.pattern_id = "existing"
        storage.load_patterns.return_value = {"existing": existing_pattern}
        result = learner.learn_from_feedback(make_operation(), make_feedback())
        assert result == "existing"
        existing_pattern.update_from_feedback.assert_called_once_with("accepted")

    def test_fingerprint_generated_when_no_hash(self):
        learner, _, fingerprinter = make_learner()
        op = make_operation(has_hash=False)
        mock_pattern = MagicMock()
        mock_pattern.pattern_id = "p2"
        with patch("refactron.patterns.learner.RefactoringPattern") as mock_cls:
            mock_cls.create.return_value = mock_pattern
            learner.learn_from_feedback(op, make_feedback())
        fingerprinter.fingerprint_code.assert_called_once()

    def test_fingerprint_failure_returns_none(self):
        learner, _, fingerprinter = make_learner()
        op = make_operation(has_hash=False)
        fingerprinter.fingerprint_code.side_effect = Exception("hash error")
        result = learner.learn_from_feedback(op, make_feedback())
        assert result is None

    def test_storage_save_failure_raises(self):
        learner, storage, _ = make_learner()
        storage.load_patterns.return_value = {}
        mock_pattern = MagicMock()
        mock_pattern.pattern_id = "p3"
        storage.save_pattern.side_effect = Exception("disk full")
        with patch("refactron.patterns.learner.RefactoringPattern") as mock_cls:
            mock_cls.create.return_value = mock_pattern
            result = learner.learn_from_feedback(make_operation(), make_feedback())
        assert result is None  # error is caught, returns None

    def test_benefit_score_updated_when_accepted_with_metric(self):
        learner, storage, _ = make_learner()
        existing_pattern = MagicMock()
        existing_pattern.pattern_hash = "h1"
        existing_pattern.operation_type = "rename"
        existing_pattern.pattern_id = "p4"
        storage.load_patterns.return_value = {"p4": existing_pattern}
        mock_metric = MagicMock()
        storage.get_pattern_metric.return_value = mock_metric
        existing_pattern.calculate_benefit_score.return_value = 0.8
        learner.learn_from_feedback(make_operation(), make_feedback(action="accepted"))
        existing_pattern.calculate_benefit_score.assert_called_once_with(mock_metric)


class TestBatchLearn:
    def test_empty_list(self):
        learner, _, _ = make_learner()
        stats = learner.batch_learn([])
        assert stats == {"processed": 0, "created": 0, "updated": 0, "failed": 0}

    def test_none_list_raises(self):
        learner, _, _ = make_learner()
        with pytest.raises(ValueError):
            learner.batch_learn(None)

    def test_skips_none_items(self):
        learner, _, _ = make_learner()
        stats = learner.batch_learn([(None, None)])
        assert stats["failed"] == 1

    def test_processes_items(self):
        learner, storage, _ = make_learner()
        mock_pattern = MagicMock()
        mock_pattern.pattern_id = "p"
        with patch("refactron.patterns.learner.RefactoringPattern") as mock_cls:
            mock_cls.create.return_value = mock_pattern
            stats = learner.batch_learn([(make_operation(), make_feedback())])
        assert stats["processed"] == 1
        assert stats["created"] == 1


class TestUpdatePatternMetrics:
    def test_empty_pattern_id_raises(self):
        learner, _, _ = make_learner()
        with pytest.raises(ValueError, match="pattern_id cannot be empty"):
            learner.update_pattern_metrics("", MagicMock(), MagicMock())

    def test_none_before_metrics_raises(self):
        learner, _, _ = make_learner()
        with pytest.raises(ValueError, match="before_metrics cannot be None"):
            learner.update_pattern_metrics("p1", None, MagicMock())

    def test_none_after_metrics_raises(self):
        learner, _, _ = make_learner()
        with pytest.raises(ValueError, match="after_metrics cannot be None"):
            learner.update_pattern_metrics("p1", MagicMock(), None)

    def test_pattern_not_found_returns_silently(self):
        learner, storage, _ = make_learner()
        storage.get_pattern.return_value = None
        learner.update_pattern_metrics("missing", MagicMock(), MagicMock())

    def test_creates_new_metric_when_none(self):
        learner, storage, _ = make_learner()
        pattern = MagicMock()
        storage.get_pattern.return_value = pattern
        storage.get_pattern_metric.return_value = None
        before = MagicMock(
            complexity=5, maintainability_index=50.0, lines_of_code=100, issue_count=3
        )
        after = MagicMock(complexity=3, maintainability_index=65.0, lines_of_code=90, issue_count=1)
        learner.update_pattern_metrics("p1", before, after)
        storage.save_pattern_metric.assert_called_once()


# ──────────────────────────── QualityGate ─────────────────────────────────────


class TestQualityGate:
    def make_result(
        self, critical=0, errors=0, warnings=0, info=0, total=0, files=5, analyzed=5, failed=0
    ):
        result = MagicMock()
        result.summary.return_value = {
            "critical": critical,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "total_issues": total,
            "total_files": files,
            "files_analyzed": analyzed,
            "files_failed": failed,
        }
        return result

    def test_passes_clean_result(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate()
        passed, msg = gate.check(self.make_result())
        assert passed is True

    def test_fails_on_critical(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(max_critical=0)
        passed, msg = gate.check(self.make_result(critical=1))
        assert passed is False

    def test_fails_on_too_many_errors(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(max_errors=5)
        passed, msg = gate.check(self.make_result(errors=6))
        assert passed is False

    def test_fails_on_too_many_warnings(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(max_warnings=10)
        passed, msg = gate.check(self.make_result(warnings=11))
        assert passed is False

    def test_fails_on_max_total(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(max_total=5)
        passed, msg = gate.check(self.make_result(total=6))
        assert passed is False

    def test_fails_on_low_success_rate(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(min_success_rate=0.95)
        passed, msg = gate.check(self.make_result(files=10, analyzed=8, failed=2))
        assert passed is False

    def test_fail_on_errors_flag(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(fail_on_errors=True)
        passed, msg = gate.check(self.make_result(errors=1))
        assert passed is False

    def test_fail_on_warnings_flag(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(fail_on_warnings=True)
        passed, msg = gate.check(self.make_result(warnings=1))
        assert passed is False

    def test_to_dict_round_trip(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate(max_critical=2, max_errors=5)
        d = gate.to_dict()
        gate2 = QualityGate.from_dict(d)
        assert gate2.max_critical == 2

    def test_no_files_skips_success_rate_check(self):
        from refactron.cicd.quality_gates import QualityGate

        gate = QualityGate()
        passed, _ = gate.check(self.make_result(files=0, analyzed=0))
        assert passed is True


class TestQualityGateParser:
    def test_parse_json_output(self, tmp_path):
        from refactron.cicd.quality_gates import QualityGateParser

        f = tmp_path / "result.json"
        f.write_text(json.dumps({"critical": 0, "errors": 1}))
        data = QualityGateParser.parse_json_output(f)
        assert data["errors"] == 1

    def test_parse_json_file_not_found(self):
        from refactron.cicd.quality_gates import QualityGateParser

        with pytest.raises(FileNotFoundError):
            QualityGateParser.parse_json_output(Path("/nonexistent/result.json"))

    def test_parse_json_invalid(self, tmp_path):
        from refactron.cicd.quality_gates import QualityGateParser

        f = tmp_path / "bad.json"
        f.write_text("not json")
        with pytest.raises(ValueError):
            QualityGateParser.parse_json_output(f)

    def test_parse_text_output(self):
        from refactron.cicd.quality_gates import QualityGateParser

        text = "Critical: 2\nErrors: 5\nWarnings: 10\nInfo: 1\nTotal: 18"
        result = QualityGateParser.parse_text_output(text)
        assert result["critical"] == 2
        assert result["errors"] == 5

    def test_parse_text_output_empty(self):
        from refactron.cicd.quality_gates import QualityGateParser

        result = QualityGateParser.parse_text_output("")
        assert result["critical"] == 0

    def test_parse_exit_code(self):
        from refactron.cicd.quality_gates import QualityGateParser

        assert QualityGateParser.parse_exit_code(0)["should_fail"] is False
        assert QualityGateParser.parse_exit_code(1)["should_fail"] is True

    def test_enforce_gate_passes(self):
        from refactron.cicd.quality_gates import QualityGate, QualityGateParser

        gate = QualityGate()
        result = MagicMock()
        result.summary.return_value = {
            "critical": 0,
            "errors": 0,
            "warnings": 0,
            "info": 0,
            "total_issues": 0,
            "total_files": 1,
            "files_analyzed": 1,
            "files_failed": 0,
        }
        passed, msg, code = QualityGateParser.enforce_gate(result, gate)
        assert passed and code == 0

    def test_generate_summary(self):
        from refactron.cicd.quality_gates import QualityGateParser

        result = MagicMock()
        result.summary.return_value = {
            "critical": 1,
            "errors": 2,
            "warnings": 3,
            "info": 0,
            "total_issues": 6,
            "total_files": 5,
            "files_analyzed": 4,
            "files_failed": 1,
        }
        summary = QualityGateParser.generate_summary(result)
        assert "Quality Gate" in summary


# ──────────────────────────── PRIntegration ───────────────────────────────────


class TestPRComment:
    def test_to_markdown_with_suggestion(self):
        from refactron.cicd.pr_integration import PRComment

        c = PRComment(
            file_path="a.py",
            line=5,
            message="Too complex",
            level="error",
            rule_id="C001",
            suggestion="x = 1",
        )
        md = c.to_markdown()
        assert "ERROR" in md and "C001" in md and "Suggestion" in md

    def test_to_markdown_no_suggestion(self):
        from refactron.cicd.pr_integration import PRComment

        c = PRComment(file_path="a.py", line=1, message="Issue", level="warning")
        md = c.to_markdown()
        assert "WARNING" in md and "Suggestion" not in md


class TestPRIntegration:
    def make_result(self, n_issues=0, n_files=2, n_critical=0, n_failed=0):
        result = MagicMock()
        result.summary.return_value = {
            "files_analyzed": n_files,
            "total_files": n_files,
            "critical": n_critical,
            "errors": 0,
            "warnings": 0,
            "info": 0,
            "total_issues": n_issues,
            "files_failed": n_failed,
        }
        result.all_issues = []
        result.file_metrics = []
        return result

    def test_generate_pr_summary_no_issues(self):
        from refactron.cicd.pr_integration import PRIntegration

        pr = PRIntegration.generate_pr_summary(self.make_result())
        assert "Refactron" in pr

    def test_generate_pr_summary_with_critical(self):
        from refactron.cicd.pr_integration import PRIntegration

        pr = PRIntegration.generate_pr_summary(self.make_result(n_critical=2, n_issues=2))
        assert "critical" in pr.lower()

    def test_generate_pr_summary_with_failed_files(self):
        from refactron.cicd.pr_integration import PRIntegration

        pr = PRIntegration.generate_pr_summary(self.make_result(n_failed=1))
        assert "failed" in pr.lower()

    def test_generate_inline_comments(self):
        from refactron.cicd.pr_integration import PRIntegration
        from refactron.core.models import IssueLevel

        result = MagicMock()
        issue = MagicMock()
        issue.level.name = "ERROR"
        issue.line_number = 5
        issue.message = "Bad code"
        issue.rule_id = "E001"
        issue.suggestion = None
        result.issues_by_file.return_value = [issue]
        comments = PRIntegration.generate_inline_comments(result, Path("a.py"))
        assert len(comments) == 1

    def test_format_comment_for_github_api(self):
        from refactron.cicd.pr_integration import PRComment, PRIntegration

        c = PRComment("a.py", 5, "msg", "error", "E001", "fix")
        data = PRIntegration.format_comment_for_github_api(c)
        assert "path" in data and "body" in data

    def test_save_comments_json(self, tmp_path):
        from refactron.cicd.pr_integration import PRComment, PRIntegration

        out = tmp_path / "comments.json"
        comments = [PRComment("a.py", 1, "msg", "warning")]
        PRIntegration.save_comments_json(comments, out)
        data = json.loads(out.read_text())
        assert len(data) == 1

    def test_generate_github_comment_body(self):
        from refactron.cicd.pr_integration import PRIntegration

        body = PRIntegration.generate_github_comment_body(self.make_result())
        assert "Refactron" in body

    def test_top_issues_by_file(self):
        from refactron.cicd.pr_integration import PRIntegration

        result = MagicMock()
        result.summary.return_value = {
            "files_analyzed": 2,
            "total_files": 2,
            "critical": 0,
            "errors": 1,
            "warnings": 0,
            "info": 0,
            "total_issues": 1,
            "files_failed": 0,
        }
        file_metrics_mock = MagicMock()
        file_metrics_mock.file_path = Path("a.py")
        file_metrics_mock.issues = [MagicMock()]
        result.all_issues = [MagicMock()]
        result.file_metrics = [file_metrics_mock]
        pr = PRIntegration.generate_pr_summary(result)
        assert "a.py" in pr


# ──────────────────────────── cli/auth.py ─────────────────────────────────────


class TestCliAuth:
    @pytest.fixture()
    def runner(self):
        return CliRunner()

    def test_login_already_logged_in_not_expired(self, runner):
        from datetime import datetime, timedelta, timezone

        from refactron.cli.auth import login

        creds = MagicMock()
        creds.access_token = "token"
        creds.email = "user@test.com"
        creds.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=creds),
        ):
            result = runner.invoke(login, [])
        assert result.exit_code == 0
        assert "Already authenticated" in result.output

    def test_login_force_relogin(self, runner):
        from refactron.cli.auth import login

        auth_result = MagicMock()
        auth_result.user_code = "CODE123"
        auth_result.device_code = "dev"
        auth_result.interval = 5
        auth_result.expires_in = 300
        token = MagicMock()
        token.access_token = "tok"
        token.token_type = "Bearer"
        token.expires_at.return_value = None
        token.email = "u@test.com"
        token.plan = "free"
        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=None),
            patch("refactron.cli.auth.start_device_authorization", return_value=auth_result),
            patch("refactron.cli.auth.poll_for_token", return_value=token),
            patch("refactron.cli.auth.save_credentials"),
            patch("refactron.cli.auth.credentials_path", return_value=Path("/tmp/creds.json")),
            patch("refactron.cli.auth._auth_banner"),
            patch("webbrowser.open"),
        ):
            result = runner.invoke(login, ["--force"])
        assert result.exit_code == 0

    def test_login_start_device_auth_fails(self, runner):
        from refactron.cli.auth import login

        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=None),
            patch(
                "refactron.cli.auth.start_device_authorization", side_effect=Exception("conn fail")
            ),
        ):
            result = runner.invoke(login, [])
        assert result.exit_code == 1

    def test_login_poll_fails(self, runner):
        from refactron.cli.auth import login

        auth_result = MagicMock()
        auth_result.user_code = "CODE"
        auth_result.device_code = "dev"
        auth_result.interval = 5
        auth_result.expires_in = 60
        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=None),
            patch("refactron.cli.auth.start_device_authorization", return_value=auth_result),
            patch("refactron.cli.auth.poll_for_token", side_effect=Exception("timeout")),
            patch("webbrowser.open"),
        ):
            result = runner.invoke(login, [])
        assert result.exit_code == 1

    def test_logout_success(self, runner):
        from refactron.cli.auth import logout

        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.credentials_path", return_value=Path("/tmp/creds.json")),
            patch("refactron.cli.auth.delete_credentials", return_value=True),
        ):
            result = runner.invoke(logout, [])
        assert "Logged Out" in result.output

    def test_logout_no_credentials(self, runner):
        from refactron.cli.auth import logout

        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.credentials_path", return_value=Path("/tmp/creds.json")),
            patch("refactron.cli.auth.delete_credentials", return_value=False),
        ):
            result = runner.invoke(logout, [])
        assert "No credentials" in result.output

    def test_auth_status_not_logged_in(self, runner):
        from refactron.cli.auth import auth_status

        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=None),
        ):
            result = runner.invoke(auth_status, [])
        assert "Not logged in" in result.output

    def test_auth_status_active(self, runner):
        from datetime import datetime, timedelta, timezone

        from refactron.cli.auth import auth_status

        creds = MagicMock()
        creds.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        creds.email = "user@test.com"
        creds.plan = "pro"
        creds.api_base_url = "https://api.refactron.dev"
        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=creds),
            patch("refactron.cli.auth._auth_banner"),
        ):
            result = runner.invoke(auth_status, [])
        assert "Active" in result.output

    def test_auth_status_expired(self, runner):
        from datetime import datetime, timedelta, timezone

        from refactron.cli.auth import auth_status

        creds = MagicMock()
        creds.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        creds.email = "user@test.com"
        creds.plan = "free"
        creds.api_base_url = "https://api.refactron.dev"
        with (
            patch("refactron.cli.auth._setup_logging"),
            patch("refactron.cli.auth.load_credentials", return_value=creds),
            patch("refactron.cli.auth._auth_banner"),
        ):
            result = runner.invoke(auth_status, [])
        assert "Expired" in result.output

    def test_telemetry_enable(self, runner):
        from refactron.cli.auth import telemetry

        with patch("refactron.cli.auth.enable_telemetry") as mock_enable:
            result = runner.invoke(telemetry, ["--enable"])
        assert result.exit_code == 0
        mock_enable.assert_called_once()

    def test_telemetry_disable(self, runner):
        from refactron.cli.auth import telemetry

        with patch("refactron.cli.auth.disable_telemetry") as mock_disable:
            result = runner.invoke(telemetry, ["--disable"])
        assert result.exit_code == 0
        mock_disable.assert_called_once()

    def test_telemetry_status_enabled(self, runner):
        from refactron.cli.auth import telemetry

        mock_collector = MagicMock()
        mock_collector.enabled = True
        with patch("refactron.cli.auth.get_telemetry_collector", return_value=mock_collector):
            result = runner.invoke(telemetry, [])
        assert "Enabled" in result.output

    def test_telemetry_status_disabled(self, runner):
        from refactron.cli.auth import telemetry

        mock_collector = MagicMock()
        mock_collector.enabled = False
        with patch("refactron.cli.auth.get_telemetry_collector", return_value=mock_collector):
            result = runner.invoke(telemetry, ["--status"])
        assert "Disabled" in result.output
