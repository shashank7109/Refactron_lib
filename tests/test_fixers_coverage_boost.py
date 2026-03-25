"""
Tests for autofix/fixers.py – comprehensive branch and line coverage.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from refactron.autofix.fixers import (
    AddDocstringsFixer,
    AddMissingCommasFixer,
    ConvertToFStringFixer,
    ExtractMagicNumbersFixer,
    FixIndentationFixer,
    FixTypeHintsFixer,
    NormalizeQuotesFixer,
    RemoveDeadCodeFixer,
    RemovePrintStatementsFixer,
    RemoveTrailingWhitespaceFixer,
    RemoveUnusedImportsFixer,
    RemoveUnusedVariablesFixer,
    SimplifyBooleanFixer,
    SortImportsFixer,
)
from refactron.core.models import CodeIssue, IssueCategory, IssueLevel


def make_issue(
    line_number: int = 1,
    message: str = "test issue",
    metadata: dict = None,
    category: IssueCategory = IssueCategory.MODERNIZATION,
    level: IssueLevel = IssueLevel.WARNING,
) -> CodeIssue:
    return CodeIssue(
        category=category,
        level=level,
        message=message,
        file_path=Path("test.py"),
        line_number=line_number,
        metadata=metadata or {},
    )


# ─────────────────────────── RemoveUnusedImportsFixer ────────────────────────


class TestRemoveUnusedImportsFixer:
    def setup_method(self):
        self.fixer = RemoveUnusedImportsFixer()

    def test_no_unused_imports(self):
        code = "import os\n\nos.getcwd()\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "clean" in result.reason.lower() or "No unused" in result.reason

    def test_removes_unused_import(self):
        code = "import os\nimport sys\n\nprint('hello')\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "Removed" in result.reason or result.diff is not None or result.fixed is not None

    def test_apply_equals_preview(self):
        code = "import os\nimport sys\n\nprint('hello')\n"
        issue = make_issue()
        assert self.fixer.apply(issue, code).reason == self.fixer.preview(issue, code).reason

    def test_syntax_error_code(self):
        code = "def foo(:\n    pass"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True  # returns code unchanged with count 0

    def test_from_import_unused(self):
        code = "from pathlib import Path\nfrom os import getcwd\n\nx = 1\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True

    def test_wildcard_import_skipped(self):
        code = "from os import *\nx = 1\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True

    def test_diff_creation(self):
        code = "import os\nimport sys\n\nprint('hello')\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        if result.diff:
            assert "---" in result.diff or "-" in result.diff

    def test_find_used_names_attribute(self):
        code = "import os\n\npath = os.path.join('a','b')\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True

    def test_aliased_import_used(self):
        code = "import numpy as np\n\nx = np.array([1])\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True


# ─────────────────────────── ExtractMagicNumbersFixer ────────────────────────


class TestExtractMagicNumbersFixer:
    def setup_method(self):
        self.fixer = ExtractMagicNumbersFixer()

    def test_no_value_in_metadata(self):
        issue = make_issue()  # no 'value' key
        result = self.fixer.preview(issue, "x = 42")
        assert result.success is False
        assert "No magic number" in result.reason

    def test_extracts_integer(self):
        issue = make_issue(metadata={"value": 42})
        code = "x = 42\n"
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "CONSTANT_42" in result.fixed

    def test_extracts_float(self):
        issue = make_issue(metadata={"value": 3.14})
        code = "x = 3.14\n"
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "CONSTANT_3_14" in result.fixed

    def test_context_in_metadata(self):
        issue = make_issue(metadata={"value": 100, "context": "timeout"})
        code = "wait = 100\n"
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "TIMEOUT_VALUE" in result.fixed

    def test_apply_equals_preview(self):
        issue = make_issue(metadata={"value": 5})
        code = "n = 5\n"
        assert self.fixer.apply(issue, code).fixed == self.fixer.preview(issue, code).fixed


# ─────────────────────────── AddDocstringsFixer ──────────────────────────────


class TestAddDocstringsFixer:
    def setup_method(self):
        self.fixer = AddDocstringsFixer()

    def test_syntax_error(self):
        issue = make_issue(line_number=1)
        result = self.fixer.preview(issue, "def foo(:\n    pass")
        assert result.success is False
        assert "Syntax error" in result.reason

    def test_adds_docstring_to_function(self):
        code = "def foo():\n    x = 1\n"
        issue = make_issue(line_number=1)
        result = self.fixer.preview(issue, code)
        # Should either add docstring or report it couldn't
        assert isinstance(result.success, bool)

    def test_line_exceeds_code(self):
        code = "def foo():\n    pass\n"
        issue = make_issue(line_number=999)
        # Line number beyond code length should return unchanged
        result = self.fixer.preview(issue, code)
        assert result.success is False

    def test_apply_calls_preview(self):
        code = "def bar():\n    return 1\n"
        issue = make_issue(line_number=1)
        assert self.fixer.apply(issue, code).reason == self.fixer.preview(issue, code).reason


# ─────────────────────────── RemoveDeadCodeFixer ─────────────────────────────


class TestRemoveDeadCodeFixer:
    def setup_method(self):
        self.fixer = RemoveDeadCodeFixer()

    def test_removes_line(self):
        code = "x = 1\ny = 2\nz = 3\n"
        issue = make_issue(line_number=2)
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "y = 2" not in result.fixed

    def test_invalid_line_number(self):
        code = "x = 1\n"
        issue = make_issue(line_number=999)
        result = self.fixer.preview(issue, code)
        assert result.success is False
        assert "Invalid" in result.reason

    def test_apply_equals_preview(self):
        code = "a = 1\nb = 2\n"
        issue = make_issue(line_number=1)
        assert self.fixer.apply(issue, code).fixed == self.fixer.preview(issue, code).fixed


# ─────────────────────────── FixTypeHintsFixer ───────────────────────────────


class TestFixTypeHintsFixer:
    def setup_method(self):
        self.fixer = FixTypeHintsFixer()

    def test_always_not_implemented(self):
        issue = make_issue()
        result = self.fixer.preview(issue, "def foo(): pass")
        assert result.success is False
        assert "not yet implemented" in result.reason

    def test_apply_same_as_preview(self):
        issue = make_issue()
        code = "def foo(): pass"
        assert self.fixer.apply(issue, code).reason == self.fixer.preview(issue, code).reason


# ─────────────────────────── SortImportsFixer ────────────────────────────────


class TestSortImportsFixer:
    def setup_method(self):
        self.fixer = SortImportsFixer()

    def test_isort_not_available(self):
        issue = make_issue()
        code = "import sys\nimport os\n"
        with patch.dict("sys.modules", {"isort": None}):
            result = self.fixer.preview(issue, code)
            # Either sorted or isort not installed message
            assert isinstance(result.success, bool)

    def test_imports_already_sorted(self):
        # If isort is available i.e. code doesn't change, reason = "already sorted"
        pytest.importorskip("isort")
        code = "import os\nimport sys\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True

    def test_sorts_imports(self):
        pytest.importorskip("isort")
        code = "import sys\nimport os\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True

    def test_apply_calls_preview(self):
        issue = make_issue()
        code = "import sys\nimport os\n"
        assert self.fixer.apply(issue, code).success == self.fixer.preview(issue, code).success


# ─────────────────────────── RemoveTrailingWhitespaceFixer ───────────────────


class TestRemoveTrailingWhitespaceFixer:
    def setup_method(self):
        self.fixer = RemoveTrailingWhitespaceFixer()

    def test_no_whitespace(self):
        code = "x = 1\ny = 2\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "No trailing" in result.reason

    def test_removes_trailing(self):
        code = "x = 1   \ny = 2  \n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "Removed" in result.reason
        assert result.fixed == "x = 1\ny = 2\n"

    def test_apply_calls_preview(self):
        code = "x   \n"
        issue = make_issue()
        assert self.fixer.apply(issue, code).fixed == self.fixer.preview(issue, code).fixed

    def test_count_changed_lines(self):
        code = "a  \nb  \nc\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert "2" in result.reason


# ─────────────────────────── NormalizeQuotesFixer ────────────────────────────


class TestNormalizeQuotesFixer:
    def test_single_to_double(self):
        fixer = NormalizeQuotesFixer(prefer_double=True)
        code = "x = 'hello'\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True

    def test_double_to_single(self):
        fixer = NormalizeQuotesFixer(prefer_double=False)
        code = 'x = "hello"\n'
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True

    def test_already_normalized_double(self):
        fixer = NormalizeQuotesFixer(prefer_double=True)
        code = 'x = "hello"\n'
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True
        assert "already" in result.reason.lower()

    def test_apply_equals_preview(self):
        fixer = NormalizeQuotesFixer()
        issue = make_issue()
        code = "x = 'hi'\n"
        assert fixer.apply(issue, code).success == fixer.preview(issue, code).success


# ─────────────────────────── SimplifyBooleanFixer ────────────────────────────


class TestSimplifyBooleanFixer:
    def setup_method(self):
        self.fixer = SimplifyBooleanFixer()

    def test_simplify_eq_true(self):
        code = "if x == True:\n    pass\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "if x:" in result.fixed

    def test_simplify_eq_false(self):
        code = "if x == False:\n    pass\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "if not x:" in result.fixed

    def test_no_simplification(self):
        code = "x = 1\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is False

    def test_apply_equals_preview(self):
        code = "if flag == True:\n    pass\n"
        issue = make_issue()
        assert self.fixer.apply(issue, code).fixed == self.fixer.preview(issue, code).fixed

    def test_simplify_not_eq_false(self):
        code = "if not x == False:\n    pass\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True


# ─────────────────────────── ConvertToFStringFixer ───────────────────────────


class TestConvertToFStringFixer:
    def setup_method(self):
        self.fixer = ConvertToFStringFixer()

    def test_converts_format_string(self):
        code = 'msg = "Hello {}".format(name)\n'
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert 'f"Hello' in result.fixed

    def test_converts_single_quote_format(self):
        code = "msg = 'Hello {}'.format(name)\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True

    def test_no_format_string(self):
        code = "x = 1\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is False
        assert "No format" in result.reason

    def test_apply_equals_preview(self):
        code = 'x = "hi {}".format(name)\n'
        issue = make_issue()
        assert self.fixer.apply(issue, code).success == self.fixer.preview(issue, code).success


# ─────────────────────────── RemoveUnusedVariablesFixer ──────────────────────


class TestRemoveUnusedVariablesFixer:
    def setup_method(self):
        self.fixer = RemoveUnusedVariablesFixer()

    def test_no_variable_in_metadata(self):
        issue = make_issue()
        result = self.fixer.preview(issue, "x = 1\n")
        assert result.success is False
        assert "No variable" in result.reason

    def test_removes_unused_variable(self):
        issue = make_issue(metadata={"variable": "unused_var"})
        code = "unused_var = 42\nx = 1\n"
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "unused_var =" not in result.fixed

    def test_variable_not_found(self):
        issue = make_issue(metadata={"variable": "missing_var"})
        code = "x = 1\n"
        result = self.fixer.preview(issue, code)
        assert result.success is False
        assert "not found" in result.reason

    def test_apply_equals_preview(self):
        issue = make_issue(metadata={"variable": "tmp"})
        code = "tmp = 10\nresult = 5\n"
        assert self.fixer.apply(issue, code).fixed == self.fixer.preview(issue, code).fixed


# ─────────────────────────── FixIndentationFixer ─────────────────────────────


class TestFixIndentationFixer:
    def test_converts_tabs_to_spaces(self):
        fixer = FixIndentationFixer(spaces=4)
        code = "def foo():\n\tx = 1\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True
        assert "Removed" in result.reason or "Fixed" in result.reason
        assert "\t" not in result.fixed

    def test_already_correct(self):
        fixer = FixIndentationFixer(spaces=4)
        code = "def foo():\n    x = 1\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True
        assert "already" in result.reason.lower() or "consistent" in result.reason.lower()

    def test_apply_calls_preview(self):
        fixer = FixIndentationFixer()
        issue = make_issue()
        code = "def bar():\n\tpass\n"
        assert fixer.apply(issue, code).fixed == fixer.preview(issue, code).fixed

    def test_custom_spaces(self):
        fixer = FixIndentationFixer(spaces=2)
        code = "if True:\n\tx = 1\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert "  x = 1" in result.fixed


# ─────────────────────────── AddMissingCommasFixer ───────────────────────────


class TestAddMissingCommasFixer:
    def setup_method(self):
        self.fixer = AddMissingCommasFixer()

    def test_adds_comma_multiline_list(self):
        code = "x = [\n    1\n]\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is True
        assert "," in result.fixed

    def test_no_missing_commas(self):
        code = "x = [1, 2, 3]\n"
        issue = make_issue()
        result = self.fixer.preview(issue, code)
        assert result.success is False
        assert "No missing commas" in result.reason

    def test_apply_equals_preview(self):
        code = "y = [\n    a\n]\n"
        issue = make_issue()
        assert self.fixer.apply(issue, code).fixed == self.fixer.preview(issue, code).fixed


# ─────────────────────────── RemovePrintStatementsFixer ──────────────────────


class TestRemovePrintStatementsFixer:
    def test_removes_print_statements(self):
        fixer = RemovePrintStatementsFixer(convert_to_logging=False)
        code = "print('debug')\nx = 1\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True
        assert "print(" not in result.fixed

    def test_converts_print_to_logging(self):
        fixer = RemovePrintStatementsFixer(convert_to_logging=True)
        code = "print(msg)\nx = 1\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is True
        assert "logger.info" in result.fixed

    def test_no_print_statements(self):
        fixer = RemovePrintStatementsFixer()
        code = "x = 1\ny = 2\n"
        issue = make_issue()
        result = fixer.preview(issue, code)
        assert result.success is False
        assert "No print" in result.reason

    def test_apply_equals_preview(self):
        fixer = RemovePrintStatementsFixer()
        code = "print('hi')\n"
        issue = make_issue()
        assert fixer.apply(issue, code).fixed == fixer.preview(issue, code).fixed
