"""Tests for llm/orchestrator.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from refactron.core.models import CodeIssue, IssueCategory, IssueLevel
from refactron.llm.models import RefactoringSuggestion, SuggestionStatus
from refactron.llm.orchestrator import LLMOrchestrator


def make_issue():
    return CodeIssue(
        category=IssueCategory.COMPLEXITY,
        level=IssueLevel.WARNING,
        message="Too complex",
        file_path=Path("test.py"),
        line_number=5,
    )


def make_orchestrator(llm_response=None, llm_raise=None, safety_pass=True):
    mock_client = MagicMock()
    if llm_raise:
        mock_client.generate.side_effect = llm_raise
    else:
        mock_client.generate.return_value = llm_response or json.dumps(
            {
                "proposed_code": "x = 1",
                "explanation": "Fixed",
                "reasoning": "ok",
                "confidence_score": 0.9,
            }
        )
    mock_client.model = "test-model"

    mock_safety = MagicMock()
    safety_result = MagicMock()
    safety_result.passed = safety_pass
    safety_result.score = 0.9 if safety_pass else 0.1
    safety_result.issues = [] if safety_pass else ["issue"]
    mock_safety.validate.return_value = safety_result

    return LLMOrchestrator(llm_client=mock_client, safety_gate=mock_safety), mock_client


class TestGenerateSuggestion:
    def test_basic_success(self):
        orch, client = make_orchestrator()
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.status == SuggestionStatus.PENDING

    def test_safety_fail_sets_rejected(self):
        orch, _ = make_orchestrator(safety_pass=False)
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.status == SuggestionStatus.REJECTED

    def test_llm_exception_returns_failed(self):
        orch, _ = make_orchestrator(llm_raise=RuntimeError("LLM down"))
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.status == SuggestionStatus.FAILED

    def test_retriever_called(self):
        orch, client = make_orchestrator()
        mock_retriever = MagicMock()
        mock_retriever.retrieve_similar.return_value = [
            MagicMock(content="context", file_path="ctx.py")
        ]
        orch.retriever = mock_retriever
        result = orch.generate_suggestion(make_issue(), "x = 0")
        mock_retriever.retrieve_similar.assert_called_once()

    def test_retriever_exception_handled(self):
        orch, _ = make_orchestrator()
        mock_retriever = MagicMock()
        mock_retriever.retrieve_similar.side_effect = Exception("rag fail")
        orch.retriever = mock_retriever
        result = orch.generate_suggestion(make_issue(), "x = 0")
        # When retriever fails, orchestrator still produces a result
        # (may be PENDING if it gracefully skips context, or FAILED if it treats it as fatal)
        assert result.status in (SuggestionStatus.PENDING, SuggestionStatus.FAILED)

    def test_confidence_string_percent(self):
        response = json.dumps(
            {
                "proposed_code": "x=1",
                "explanation": "ok",
                "reasoning": "r",
                "confidence_score": "85%",
            }
        )
        orch, _ = make_orchestrator(llm_response=response)
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.status in (SuggestionStatus.PENDING, SuggestionStatus.REJECTED)

    def test_invalid_confidence_falls_back(self):
        response = json.dumps(
            {"proposed_code": "x=1", "explanation": "ok", "confidence_score": "invalid"}
        )
        orch, _ = make_orchestrator(llm_response=response)
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.confidence_score > 0

    def test_safety_validation_exception_sets_failed(self):
        orch, _ = make_orchestrator()
        orch.safety_gate.validate.side_effect = Exception("safety crash")
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.status == SuggestionStatus.FAILED

    def test_markdown_code_block_stripped(self):
        response = (
            "```json\n"
            + json.dumps({"proposed_code": "x=1", "explanation": "ok", "confidence_score": 0.9})
            + "\n```"
        )
        orch, _ = make_orchestrator(llm_response=response)
        result = orch.generate_suggestion(make_issue(), "x = 0")
        assert result.status != SuggestionStatus.FAILED


class TestGenerateDocumentation:
    def test_success_with_delimiters(self):
        response = "@@@EXPLANATION@@@Added docstrings.@@@CONFIDENCE@@@0.85@@@MARKDOWN@@@# Docs\n"
        orch, _ = make_orchestrator(llm_response=response)
        result = orch.generate_documentation("def foo(): pass")
        assert result.status == SuggestionStatus.PENDING

    def test_no_delimiter_raises_valueerror(self):
        orch, _ = make_orchestrator(llm_response="just plain text")
        result = orch.generate_documentation("def foo(): pass")
        assert result.status == SuggestionStatus.FAILED

    def test_llm_exception_returns_failed(self):
        orch, _ = make_orchestrator(llm_raise=RuntimeError("fail"))
        result = orch.generate_documentation("def foo(): pass")
        assert result.status == SuggestionStatus.FAILED

    def test_invalid_confidence_falls_back(self):
        response = "@@@EXPLANATION@@@ok@@@CONFIDENCE@@@BAD@@@MARKDOWN@@@# x\n"
        orch, _ = make_orchestrator(llm_response=response)
        result = orch.generate_documentation("def foo(): pass")
        assert result.confidence_score == pytest.approx(0.8)


class TestEvaluateIssuesBatch:
    def test_empty_issues(self):
        orch, _ = make_orchestrator()
        result = orch.evaluate_issues_batch([], "x=1")
        assert result == {}

    def test_successful_batch(self):
        orch, client = make_orchestrator()
        client.generate.return_value = json.dumps({"issue:1:0": 0.9})
        issues = [make_issue()]
        result = orch.evaluate_issues_batch(issues, "x=1")
        assert isinstance(result, dict)

    def test_llm_exception_returns_defaults(self):
        orch, client = make_orchestrator()
        client.generate.side_effect = RuntimeError("fail")
        issues = [make_issue()]
        result = orch.evaluate_issues_batch(issues, "x=1")
        assert all(v == 0.5 for v in result.values())

    def test_duplicate_issue_ids(self):
        orch, client = make_orchestrator()
        client.generate.return_value = json.dumps({})
        issue1 = make_issue()
        issue2 = make_issue()  # same attributes → potential ID collision
        result = orch.evaluate_issues_batch([issue1, issue2], "x=1")
        assert isinstance(result, dict)


class TestCleanJsonResponse:
    def setup_method(self):
        self.orch, _ = make_orchestrator()

    def test_plain_json(self):
        assert self.orch._clean_json_response('{"a":1}') == '{"a":1}'

    def test_markdown_json_block(self):
        text = '```json\n{"a":1}\n```'
        assert self.orch._clean_json_response(text) == '{"a":1}'

    def test_generic_code_block(self):
        text = '```\n{"a":1}\n```'
        assert self.orch._clean_json_response(text) == '{"a":1}'

    def test_unclosed_code_block(self):
        text = '```json\n{"a":1}'
        result = self.orch._clean_json_response(text)
        assert '{"a":1}' in result


class TestOrchestratorInit:
    def test_uses_groq_if_env_set(self):
        mock_groq = MagicMock()
        with patch.dict("os.environ", {"GROQ_API_KEY": "key"}), patch(
            "refactron.llm.orchestrator.GroqClient", return_value=mock_groq
        ):
            orch = LLMOrchestrator()
        assert orch.client is mock_groq

    def test_falls_back_to_backend_if_groq_fails(self):
        mock_backend = MagicMock()
        with patch.dict("os.environ", {"GROQ_API_KEY": "key"}), patch(
            "refactron.llm.orchestrator.GroqClient", side_effect=RuntimeError
        ), patch("refactron.llm.orchestrator.BackendLLMClient", return_value=mock_backend):
            orch = LLMOrchestrator()
        assert orch.client is mock_backend

    def test_uses_backend_without_env(self):
        mock_backend = MagicMock()
        with patch.dict("os.environ", {}, clear=True), patch(
            "refactron.llm.orchestrator.BackendLLMClient", return_value=mock_backend
        ):
            orch = LLMOrchestrator()
        assert orch.client is mock_backend
