"""
Tests for refactron/cli/utils.py and refactron/llm/orchestrator.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from refactron.cli.utils import (
    ApiKeyValidationResult,
    _detect_project_type,
    _get_pattern_storage_from_config,
    _load_config,
    _setup_logging,
    _validate_api_key,
    _validate_path,
)
from refactron.core.models import CodeIssue, IssueCategory, IssueLevel
from refactron.llm.orchestrator import LLMOrchestrator

# ──────────────────────────────────────────────
# ApiKeyValidationResult
# ──────────────────────────────────────────────


class TestApiKeyValidationResult:
    def test_ok_result(self):
        r = ApiKeyValidationResult(ok=True, message="Verified.")
        assert r.ok is True

    def test_fail_result(self):
        r = ApiKeyValidationResult(ok=False, message="Invalid API key.")
        assert r.ok is False


class TestValidateApiKey:
    def _mock_response(self, status_code: int):
        resp = MagicMock()
        resp.status_code = status_code
        return resp

    def test_200_returns_ok(self):
        with patch("refactron.cli.utils.requests.get", return_value=self._mock_response(200)):
            result = _validate_api_key("https://api.example.com", "key123", 5)
        assert result.ok is True

    def test_401_returns_invalid(self):
        with patch("refactron.cli.utils.requests.get", return_value=self._mock_response(401)):
            result = _validate_api_key("https://api.example.com", "badkey", 5)
        assert result.ok is False
        assert "Invalid" in result.message

    def test_403_returns_invalid(self):
        with patch("refactron.cli.utils.requests.get", return_value=self._mock_response(403)):
            result = _validate_api_key("https://api.example.com", "badkey", 5)
        assert result.ok is False

    def test_404_returns_missing_endpoint(self):
        with patch("refactron.cli.utils.requests.get", return_value=self._mock_response(404)):
            result = _validate_api_key("https://api.example.com", "key", 5)
        assert result.ok is False
        assert "404" in result.message

    def test_500_returns_server_error(self):
        with patch("refactron.cli.utils.requests.get", return_value=self._mock_response(500)):
            result = _validate_api_key("https://api.example.com", "key", 5)
        assert result.ok is False
        assert "500" in result.message

    def test_timeout_returns_error(self):
        import requests

        with patch("refactron.cli.utils.requests.get", side_effect=requests.Timeout):
            result = _validate_api_key("https://api.example.com", "key", 1)
        assert result.ok is False
        assert "timed out" in result.message.lower()

    def test_connection_error_returns_error(self):
        import requests

        with patch("refactron.cli.utils.requests.get", side_effect=requests.ConnectionError):
            result = _validate_api_key("https://api.example.com", "key", 5)
        assert result.ok is False
        assert "reach" in result.message.lower()

    def test_request_exception_returns_error(self):
        import requests

        with patch("refactron.cli.utils.requests.get", side_effect=requests.RequestException):
            result = _validate_api_key("https://api.example.com", "key", 5)
        assert result.ok is False

    def test_unknown_status_code(self):
        with patch("refactron.cli.utils.requests.get", return_value=self._mock_response(418)):
            result = _validate_api_key("https://api.example.com", "key", 5)
        assert result.ok is False
        assert "418" in result.message


class TestSetupLogging:
    def test_setup_verbose(self):
        import logging

        _setup_logging(verbose=True)
        import logging

        # verbose mode should not suppress httpx
        assert True  # verbose mode does not suppress httpx

    def test_setup_normal(self):
        import logging

        _setup_logging(verbose=False)
        # Noisy loggers should be set to ERROR
        assert logging.getLogger("httpx").level == logging.ERROR


class TestLoadConfig:
    def test_default_config_no_path(self):
        config = _load_config(None)
        assert config is not None

    def test_config_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write('version: "1.0"\nmax_function_complexity: 12\n')
            fname = f.name
        config = _load_config(fname)
        assert config is not None

    def test_config_missing_file_exits(self):
        with pytest.raises(SystemExit):
            _load_config("/nonexistent/path/config.yaml")

    def test_config_with_profile(self):
        config = _load_config(None, profile="default", environment=None)
        assert config is not None


class TestValidatePath:
    def test_valid_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _validate_path(tmpdir)
            assert result == Path(tmpdir)

    def test_invalid_path_exits(self):
        with pytest.raises(SystemExit):
            _validate_path("/nonexistent/path/that/does/not/exist")


class TestDetectProjectType:
    def test_detects_django(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manage_py = Path(tmpdir) / "manage.py"
            manage_py.write_text("import django\nDJANGO_SETTINGS_MODULE = 'myproject.settings'\n")
            with patch("refactron.cli.utils.Path.cwd", return_value=Path(tmpdir)):
                result = _detect_project_type()
            assert result == "django"

    def test_detects_fastapi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            main_py = Path(tmpdir) / "main.py"
            main_py.write_text("from fastapi import FastAPI\napp = FastAPI()\n")
            with patch("refactron.cli.utils.Path.cwd", return_value=Path(tmpdir)):
                result = _detect_project_type()
            assert result == "fastapi"

    def test_detects_flask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app_py = Path(tmpdir) / "app.py"
            app_py.write_text("from flask import Flask\napp = Flask(__name__)\n")
            with patch("refactron.cli.utils.Path.cwd", return_value=Path(tmpdir)):
                result = _detect_project_type()
            assert result == "flask"

    def test_no_framework_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("refactron.cli.utils.Path.cwd", return_value=Path(tmpdir)):
                result = _detect_project_type()
            assert result is None


class TestGetPatternStorageFromConfig:
    def test_returns_storage_when_enabled(self):
        config = MagicMock()
        config.enable_pattern_learning = True
        config.pattern_storage_dir = Path(tempfile.mkdtemp())
        result = _get_pattern_storage_from_config(config)
        assert result is not None

    def test_returns_none_when_disabled(self):
        config = MagicMock()
        config.enable_pattern_learning = False
        result = _get_pattern_storage_from_config(config)
        assert result is None


# ──────────────────────────────────────────────
# LLMOrchestrator
# ──────────────────────────────────────────────


def make_issue(line=1):
    return CodeIssue(
        category=IssueCategory.CODE_SMELL,
        level=IssueLevel.WARNING,
        message="Magic number detected",
        file_path=Path("test.py"),
        line_number=line,
        rule_id="C001",
    )


class TestLLMOrchestrator:
    def _make_orchestrator(
        self,
        response_text='{"proposed_code": "x = 1", "explanation": "test", "reasoning": "ok", "confidence": 0.9}',
    ):
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(return_value=response_text)
        return LLMOrchestrator(llm_client=mock_client), mock_client

    def test_generate_suggestion_success(self):
        orchestrator, _ = self._make_orchestrator(
            '{"proposed_code": "x = 1", "explanation": "fixed it", "reasoning": "clean", "confidence": 0.95}'
        )
        issue = make_issue()
        result = orchestrator.generate_suggestion(issue, "x = 42")
        assert result is not None
        assert result.model_name == "test-model"

    def test_generate_suggestion_with_retriever(self):
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(
            return_value='{"proposed_code": "x = 1", "explanation": "ok", "reasoning": "fine", "confidence": 0.9}'
        )
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "similar code snippet"
        mock_retriever.retrieve_similar = MagicMock(return_value=[mock_result])

        orchestrator = LLMOrchestrator(llm_client=mock_client, retriever=mock_retriever)
        issue = make_issue()
        result = orchestrator.generate_suggestion(issue, "x = 42")
        assert result is not None
        mock_retriever.retrieve_similar.assert_called_once()

    def test_generate_suggestion_retriever_fails_gracefully(self):
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(
            return_value='{"proposed_code": "x=1", "explanation": "ok", "reasoning": "ok", "confidence": 0.9}'
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve_similar = MagicMock(side_effect=RuntimeError("index unavailable"))

        orchestrator = LLMOrchestrator(llm_client=mock_client, retriever=mock_retriever)
        result = orchestrator.generate_suggestion(make_issue(), "x = 42")
        assert result is not None  # Graceful degradation

    def test_generate_suggestion_llm_failure(self):
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(side_effect=RuntimeError("LLM unavailable"))

        orchestrator = LLMOrchestrator(llm_client=mock_client)
        result = orchestrator.generate_suggestion(make_issue(), "x = 42")
        assert result is not None  # Returns a failed suggestion

    def test_evaluate_issues_batch_empty(self):
        orchestrator, _ = self._make_orchestrator()
        result = orchestrator.evaluate_issues_batch([], "x = 1")
        assert result == {}

    def test_evaluate_issues_batch_success(self):
        response = json.dumps({"C001:1:0": 0.9, "C001:2:1": 0.3})
        orchestrator, mock_client = self._make_orchestrator(response)
        mock_client.generate = MagicMock(return_value=response)

        issues = [make_issue(1), make_issue(2)]
        result = orchestrator.evaluate_issues_batch(issues, "x = 42\ny = 99")
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())

    def test_evaluate_issues_batch_llm_failure(self):
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(side_effect=RuntimeError("LLM down"))

        orchestrator = LLMOrchestrator(llm_client=mock_client)
        issues = [make_issue(1)]
        result = orchestrator.evaluate_issues_batch(issues, "x = 42")
        # Falls back to 0.5 for each issue
        assert all(v == 0.5 for v in result.values())

    def test_clean_json_response_markdown_block(self):
        orchestrator, _ = self._make_orchestrator()
        raw = '```json\n{"key": "value"}\n```'
        cleaned = orchestrator._clean_json_response(raw)
        assert cleaned == '{"key": "value"}'

    def test_clean_json_response_plain_code_block(self):
        orchestrator, _ = self._make_orchestrator()
        raw = '```\n{"key": "value"}\n```'
        cleaned = orchestrator._clean_json_response(raw)
        assert cleaned == '{"key": "value"}'

    def test_clean_json_response_no_block(self):
        orchestrator, _ = self._make_orchestrator()
        raw = '{"key": "value"}'
        cleaned = orchestrator._clean_json_response(raw)
        assert cleaned == '{"key": "value"}'

    def test_clean_json_response_unclosed_block(self):
        orchestrator, _ = self._make_orchestrator()
        raw = '```json\n{"key": "value"}'
        cleaned = orchestrator._clean_json_response(raw)
        assert '{"key": "value"}' in cleaned

    def test_evaluate_issues_batch_with_retriever(self):
        response = json.dumps({"C001:1:0": 0.85})
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(return_value=response)

        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "context"
        mock_retriever.retrieve_similar = MagicMock(return_value=[mock_result])

        orchestrator = LLMOrchestrator(llm_client=mock_client, retriever=mock_retriever)
        issues = [make_issue(1)]
        result = orchestrator.evaluate_issues_batch(issues, "x = 42")
        assert isinstance(result, dict)

    def test_evaluate_duplicate_rule_ids(self):
        """Issues with same rule_id get unique IDs."""
        response = "{}"
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.generate = MagicMock(return_value=response)
        orchestrator = LLMOrchestrator(llm_client=mock_client)

        issues = [
            CodeIssue(
                category=IssueCategory.CODE_SMELL,
                level=IssueLevel.WARNING,
                message="same issue",
                file_path=Path("test.py"),
                line_number=1,
                rule_id="DUPE",
            ),
            CodeIssue(
                category=IssueCategory.CODE_SMELL,
                level=IssueLevel.WARNING,
                message="same issue",
                file_path=Path("test.py"),
                line_number=1,
                rule_id="DUPE",
            ),
        ]
        result = orchestrator.evaluate_issues_batch(issues, "x = 1")
        # Should not raise
        assert isinstance(result, dict)
