"""Tests for llm/safety.py"""

from unittest.mock import MagicMock

import pytest

from refactron.llm.models import RefactoringSuggestion, SuggestionStatus
from refactron.llm.safety import SafetyGate


def make_suggestion(proposed="x = 1", original="x = 0", confidence=0.9):
    s = MagicMock(spec=RefactoringSuggestion)
    s.proposed_code = proposed
    s.original_code = original
    s.confidence_score = confidence
    return s


class TestSafetyGateValidate:
    def setup_method(self):
        self.gate = SafetyGate(min_confidence=0.7)

    def test_clean_code_passes(self):
        s = make_suggestion("x = 1")
        result = self.gate.validate(s)
        assert result.passed is True
        assert result.syntax_valid is True
        assert result.score > 0.8

    def test_syntax_error_fails(self):
        s = make_suggestion("def foo(:\n    pass")
        result = self.gate.validate(s)
        assert result.passed is False
        assert result.syntax_valid is False
        assert result.score == 0.0

    def test_low_confidence_reduces_score(self):
        s = make_suggestion(confidence=0.3)
        result = self.gate.validate(s)
        assert result.score < 1.0
        assert any("Low confidence" in i for i in result.issues)

    def test_risky_keyword_reduces_score(self):
        s = make_suggestion("import subprocess\nsubprocess.run('ls')")
        result = self.gate.validate(s)
        assert result.score < 1.0

    def test_dangerous_new_import_flagged(self):
        s = make_suggestion(proposed="import os\nos.remove('file')", original="x = 1")
        result = self.gate.validate(s)
        assert "os" in " ".join(result.side_effects)

    def test_existing_import_not_flagged(self):
        s = make_suggestion(proposed="import os\nos.remove('file')", original="import os\n")
        result = self.gate.validate(s)
        assert not any("Import: os" in e for e in result.side_effects)

    def test_multiple_risky_keywords(self):
        code = "import subprocess\nos.system('rm')\neval('x')"
        s = make_suggestion(code)
        result = self.gate.validate(s)
        assert result.score < 1.0


class TestAssessRisk:
    def setup_method(self):
        self.gate = SafetyGate()

    def test_no_risk(self):
        assert self.gate._assess_risk("x = 1") == 0.0

    def test_max_risk_capped_at_1(self):
        code = "subprocess.run(); os.system(); eval(); exec(); shutil.rmtree(); requests.get(); urllib.open('x'); open('f')"
        assert self.gate._assess_risk(code) <= 1.0

    def test_single_risky_keyword(self):
        assert self.gate._assess_risk("eval('x')") == pytest.approx(0.3)


class TestCheckDangerousImports:
    def setup_method(self):
        self.gate = SafetyGate()

    def test_no_dangerous_imports(self):
        result = self.gate._check_dangerous_imports("x = 1", "x = 1")
        assert result == []

    def test_new_sys_import(self):
        result = self.gate._check_dangerous_imports("import sys\n", "x = 1")
        assert "sys" in result

    def test_from_import_dangerous(self):
        result = self.gate._check_dangerous_imports("from shutil import rmtree\n", "x = 1")
        assert "shutil" in result

    def test_syntax_error_in_code(self):
        result = self.gate._check_dangerous_imports("def foo(:\n    pass", "x = 1")
        assert result == []
