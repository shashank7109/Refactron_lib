"""Tests for LLM Orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from refactron.core.models import CodeIssue, IssueCategory, IssueLevel
from refactron.llm.models import SuggestionStatus
from refactron.llm.orchestrator import LLMOrchestrator
from refactron.rag.retriever import RetrievedContext


class TestLLMOrchestrator:
    """Tests for LLMOrchestrator."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = Mock()
        retriever.retrieve_similar.return_value = []
        return retriever

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.model = "llama-3.3-70b-versatile"
        # Return valid JSON in code block
        client.generate.return_value = """
```json
{
    "proposed_code": "def fixed(): pass",
    "explanation": "Fixed the bug",
    "reasoning": "Because it was broken"
}
```
"""
        return client

    @pytest.fixture
    def mock_safety(self):
        gate = Mock()
        result = MagicMock()
        result.passed = True
        result.issues = []
        gate.validate.return_value = result
        return gate

    @pytest.fixture
    def sample_issue(self):
        return CodeIssue(
            category=IssueCategory.CODE_SMELL,
            level=IssueLevel.WARNING,
            message="Function is too long",
            file_path=Path("/test.py"),
            line_number=10,
        )

    def test_generate_suggestion_basic(self, mock_client, mock_safety, sample_issue):
        """Test generating a successful suggestion."""
        orchestrator = LLMOrchestrator(llm_client=mock_client, safety_gate=mock_safety)

        suggestion = orchestrator.generate_suggestion(
            issue=sample_issue, original_code="def broken(): pass"
        )

        assert suggestion.status == SuggestionStatus.PENDING
        assert suggestion.proposed_code == "def fixed(): pass"
        assert suggestion.explanation == "Fixed the bug"
        assert suggestion.model_name == "llama-3.3-70b-versatile"

        # Verify prompt construction (implicity) by checking generate call
        mock_client.generate.assert_called_once()
        args = mock_client.generate.call_args
        assert "Function is too long" in args.kwargs["prompt"]
        assert "def broken(): pass" in args.kwargs["prompt"]

    def test_generate_suggestion_with_rag(self, mock_client, mock_retriever, sample_issue):
        """Test generation with RAG context."""
        # Setup retriever to return context
        mock_retriever.retrieve_similar.return_value = [
            RetrievedContext(
                content="def similar_func(): pass",
                file_path="/similar.py",
                chunk_type="function",
                name="similar_func",
                line_range=(1, 2),
                distance=0.1,
                metadata={},
            )
        ]

        orchestrator = LLMOrchestrator(retriever=mock_retriever, llm_client=mock_client)

        suggestion = orchestrator.generate_suggestion(issue=sample_issue, original_code="original")

        assert "/similar.py" in suggestion.context_files

        # Verify prompt contains RAG context
        args = mock_client.generate.call_args
        assert "def similar_func(): pass" in args.kwargs["prompt"]

    def test_bad_llm_response(self, mock_client, sample_issue):
        """Test handling of invalid JSON from LLM."""
        mock_client.generate.return_value = "This is not JSON"

        orchestrator = LLMOrchestrator(llm_client=mock_client)

        suggestion = orchestrator.generate_suggestion(issue=sample_issue, original_code="original")

        assert suggestion.status == SuggestionStatus.FAILED
        assert (
            "JSONDecodeError" in suggestion.explanation
            or "Expecting value" in suggestion.explanation
        )

    def test_safety_check_failure(self, mock_client, mock_safety, sample_issue):
        """Test rejection when safety check fails."""
        # Setup safety failure
        result = MagicMock()
        result.passed = False
        result.issues = ["Syntax Error"]
        mock_safety.validate.return_value = result

        orchestrator = LLMOrchestrator(llm_client=mock_client, safety_gate=mock_safety)

        suggestion = orchestrator.generate_suggestion(issue=sample_issue, original_code="original")

        assert suggestion.status == SuggestionStatus.REJECTED
        assert not suggestion.safety_result.passed
