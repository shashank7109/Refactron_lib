"""Tests for analyzers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from refactron.analyzers.code_smell_analyzer import CodeSmellAnalyzer
from refactron.analyzers.complexity_analyzer import ComplexityAnalyzer
from refactron.core.config import RefactronConfig
from refactron.core.models import IssueLevel
from refactron.llm.orchestrator import LLMOrchestrator


def test_complexity_analyzer() -> None:
    """Test complexity analyzer."""
    config = RefactronConfig(max_function_complexity=5)
    analyzer = ComplexityAnalyzer(config)

    code = """
def complex_function(x, y, z):
    if x > 0:
        if y > 10:
            if z > 20:
                if x > 30:
                    if y > 40:
                        if z > 50:
                            return "very high"
                        return "high"
                    return "medium"
                return "low"
            return "very low"
        return "negative"
    return "zero"
"""

    issues = analyzer.analyze(Path("test.py"), code)
    # Should detect high complexity
    assert len(issues) > 0
    assert analyzer.name == "complexity"


def test_code_smell_analyzer() -> None:
    """Test code smell analyzer."""
    config = RefactronConfig()
    analyzer = CodeSmellAnalyzer(config)

    code = """
def function_with_many_params(a, b, c, d, e, f, g, h):
    return a + b + c + d + e + f + g + h

class MyClass:
    def method_without_docstring(self):
        magic_number = 12345
        return magic_number
"""

    issues = analyzer.analyze(Path("test.py"), code)
    assert len(issues) > 0
    assert analyzer.name == "code_smells"


def test_analyzer_handles_syntax_errors() -> None:
    """Test that analyzers handle syntax errors gracefully."""
    config = RefactronConfig()
    analyzer = ComplexityAnalyzer(config)

    # Invalid Python code
    code = "def broken function(:"

    issues = analyzer.analyze(Path("test.py"), code)
    # Should return an error issue, not crash
    assert len(issues) >= 0


# ---------------------------------------------------------------------------
# AI triage integration tests
# ---------------------------------------------------------------------------

_SMELLY_CODE = """
def function_with_many_params(a, b, c, d, e, f, g, h):
    magic = 12345
    return a + b + c + d + e + f + g + h
"""


@pytest.fixture
def mock_orchestrator():
    return MagicMock(spec=LLMOrchestrator)


def test_ai_triage_disabled_by_default(mock_orchestrator) -> None:
    """Orchestrator is never called when enable_ai_triage is False."""
    config = RefactronConfig(enable_ai_triage=False)
    analyzer = CodeSmellAnalyzer(config, orchestrator=mock_orchestrator)

    issues = analyzer.analyze(Path("test.py"), _SMELLY_CODE)

    mock_orchestrator.evaluate_issues_batch.assert_not_called()
    assert len(issues) > 0


def _make_scores(issues: list, confidence_fn) -> dict:
    """Build a scores dict using the same ID formula as _build_issue_id."""
    return {
        CodeSmellAnalyzer._build_issue_id(issue, i): confidence_fn(i)
        for i, issue in enumerate(issues)
    }


def test_ai_triage_filters_low_confidence_issues(mock_orchestrator) -> None:
    """Issues with confidence < 0.3 are dropped when AI triage is enabled."""
    config = RefactronConfig(enable_ai_triage=True)
    analyzer = CodeSmellAnalyzer(config, orchestrator=mock_orchestrator)

    # Run once without triage to determine the triageable (non-ERROR) issues
    raw_issues = CodeSmellAnalyzer(RefactronConfig()).analyze(Path("test.py"), _SMELLY_CODE)
    triageable = [i for i in raw_issues if i.level != IssueLevel.ERROR]
    assert len(triageable) >= 2, "fixture must produce at least 2 triageable issues"

    # First issue is a true positive (high confidence), rest are false positives
    scores = _make_scores(triageable, lambda i: 0.9 if i == 0 else 0.1)
    mock_orchestrator.evaluate_issues_batch.return_value = scores

    filtered = analyzer.analyze(Path("test.py"), _SMELLY_CODE)

    mock_orchestrator.evaluate_issues_batch.assert_called_once()
    # Only the one issue above the 0.3 threshold should survive
    assert len(filtered) == 1


def test_ai_triage_keeps_all_high_confidence_issues(mock_orchestrator) -> None:
    """All issues are retained when every confidence score is >= 0.3."""
    config = RefactronConfig(enable_ai_triage=True)
    analyzer = CodeSmellAnalyzer(config, orchestrator=mock_orchestrator)

    raw_issues = CodeSmellAnalyzer(RefactronConfig()).analyze(Path("test.py"), _SMELLY_CODE)
    triageable = [i for i in raw_issues if i.level != IssueLevel.ERROR]
    assert len(triageable) >= 1

    scores = _make_scores(triageable, lambda i: 0.85)
    mock_orchestrator.evaluate_issues_batch.return_value = scores

    filtered = analyzer.analyze(Path("test.py"), _SMELLY_CODE)

    assert len(filtered) == len(raw_issues)


def test_ai_triage_passes_source_code_to_orchestrator(mock_orchestrator) -> None:
    """evaluate_issues_batch receives the full source_code string."""
    config = RefactronConfig(enable_ai_triage=True)
    analyzer = CodeSmellAnalyzer(config, orchestrator=mock_orchestrator)

    raw_issues = CodeSmellAnalyzer(RefactronConfig()).analyze(Path("test.py"), _SMELLY_CODE)
    triageable = [i for i in raw_issues if i.level != IssueLevel.ERROR]
    scores = _make_scores(triageable, lambda i: 0.9)
    mock_orchestrator.evaluate_issues_batch.return_value = scores

    analyzer.analyze(Path("test.py"), _SMELLY_CODE)

    call_args = mock_orchestrator.evaluate_issues_batch.call_args
    assert call_args[0][1] == _SMELLY_CODE


def test_ai_triage_skipped_when_no_issues(mock_orchestrator) -> None:
    """Orchestrator is not invoked when the static analysis finds nothing."""
    config = RefactronConfig(enable_ai_triage=True)
    analyzer = CodeSmellAnalyzer(config, orchestrator=mock_orchestrator)

    # Perfectly clean, trivial code
    clean_code = 'def clean() -> None:\n    """No issues here."""\n'

    analyzer.analyze(Path("test.py"), clean_code)

    mock_orchestrator.evaluate_issues_batch.assert_not_called()


def test_ai_triage_returns_all_issues_on_orchestrator_exception(mock_orchestrator) -> None:
    """If the orchestrator raises, every issue returned must match the untriaged baseline."""
    file_path = Path("test.py")

    # Establish a ground-truth baseline with AI triage completely off
    baseline_issues = CodeSmellAnalyzer(RefactronConfig(enable_ai_triage=False)).analyze(
        file_path, _SMELLY_CODE
    )
    assert len(baseline_issues) > 0, "fixture must produce at least one issue"

    # Now run with triage enabled but a crashing orchestrator
    mock_orchestrator.evaluate_issues_batch.side_effect = RuntimeError("LLM unavailable")
    config = RefactronConfig(enable_ai_triage=True)
    analyzer = CodeSmellAnalyzer(config, orchestrator=mock_orchestrator)

    fallback_issues = analyzer.analyze(file_path, _SMELLY_CODE)

    # Count must be identical — nothing was filtered
    assert len(fallback_issues) == len(baseline_issues)

    # Contents must be identical — same messages, same lines, same rule IDs
    baseline_keys = {(i.message, i.line_number, i.rule_id) for i in baseline_issues}
    fallback_keys = {(i.message, i.line_number, i.rule_id) for i in fallback_issues}
    assert fallback_keys == baseline_keys
