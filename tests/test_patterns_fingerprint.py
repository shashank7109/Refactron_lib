"""Tests for PatternFingerprinter."""

import shutil
import tempfile
from pathlib import Path

from refactron.core.models import CodeIssue, IssueCategory, IssueLevel, RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter


class TestPatternFingerprinter:
    """Test cases for PatternFingerprinter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fingerprinter = PatternFingerprinter()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fingerprint_code_basic(self):
        """Test basic code fingerprinting."""
        code = "def hello():\n    print('world')"
        hash1 = self.fingerprinter.fingerprint_code(code)
        hash2 = self.fingerprinter.fingerprint_code(code)

        # Same code should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 char hex string

    def test_fingerprint_code_consistent(self):
        """Test that fingerprinting is consistent across calls."""
        code1 = "def add(a, b):\n    return a + b"
        code2 = "def add(a, b):\n    return a + b"
        hash1 = self.fingerprinter.fingerprint_code(code1)
        hash2 = self.fingerprinter.fingerprint_code(code2)

        assert hash1 == hash2

    def test_fingerprint_code_different_code(self):
        """Test that different code produces different hashes."""
        code1 = "def add(a, b):\n    return a + b"
        code2 = "def subtract(a, b):\n    return a - b"
        hash1 = self.fingerprinter.fingerprint_code(code1)
        hash2 = self.fingerprinter.fingerprint_code(code2)

        assert hash1 != hash2

    def test_fingerprint_code_whitespace_insensitive(self):
        """Test that whitespace differences don't affect fingerprint."""
        code1 = "def add(a, b):\n    return a + b"
        code2 = "def add(a,b):\n    return a+b"
        hash1 = self.fingerprinter.fingerprint_code(code1)
        hash2 = self.fingerprinter.fingerprint_code(code2)

        # Should produce same hash because normalization anonymizes code
        assert hash1 == hash2  # Comment updated to match actual behavior

    def test_fingerprint_code_comment_removal(self):
        """Test that comments are removed before fingerprinting."""
        code1 = "def add(a, b):\n    return a + b  # Add two numbers"
        code2 = "def add(a, b):\n    return a + b  # Different comment"
        hash1 = self.fingerprinter.fingerprint_code(code1)
        hash2 = self.fingerprinter.fingerprint_code(code2)

        # Should be same after comment removal
        assert hash1 == hash2

    def test_fingerprint_issue_context(self):
        """Test fingerprinting issue context."""
        source_code = (
            "def function1():\n    pass\n\n"
            "def function2():\n    x = 1\n    y = 2\n    z = x + y\n    return z"
        )
        issue = CodeIssue(
            category=IssueCategory.CODE_SMELL,
            level=IssueLevel.WARNING,
            message="Too many variables",
            file_path=Path("test.py"),
            line_number=6,
            rule_id="S001",
        )

        hash1 = self.fingerprinter.fingerprint_issue_context(issue, source_code)
        hash2 = self.fingerprinter.fingerprint_issue_context(issue, source_code)

        # Same issue context should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_fingerprint_issue_context_different_issues(self):
        """Test that different issues produce different hashes."""
        source_code = (
            "def function1():\n    pass\n\n"
            "def function2():\n    x = 1\n    y = 2\n    z = x + y\n    return z"
        )
        issue1 = CodeIssue(
            category=IssueCategory.CODE_SMELL,
            level=IssueLevel.WARNING,
            message="Too many variables",
            file_path=Path("test.py"),
            line_number=6,
            rule_id="S001",
        )
        issue2 = CodeIssue(
            category=IssueCategory.PERFORMANCE,
            level=IssueLevel.ERROR,
            message="Inefficient loop",
            file_path=Path("test.py"),
            line_number=6,
            rule_id="P001",
        )

        hash1 = self.fingerprinter.fingerprint_issue_context(issue1, source_code)
        hash2 = self.fingerprinter.fingerprint_issue_context(issue2, source_code)

        # Different issues should produce different hashes
        assert hash1 != hash2

    def test_fingerprint_refactoring(self):
        """Test fingerprinting refactoring operation."""
        operation = RefactoringOperation(
            operation_type="extract_method",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract method",
            old_code="x = 1\ny = 2\nz = x + y",
            new_code="z = add(1, 2)",
            risk_score=0.3,
        )

        hash1 = self.fingerprinter.fingerprint_refactoring(operation)
        hash2 = self.fingerprinter.fingerprint_refactoring(operation)

        # Same operation should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_fingerprint_refactoring_different_operations(self):
        """Test that different operations produce different hashes."""
        operation1 = RefactoringOperation(
            operation_type="extract_method",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract method",
            old_code="x = 1\ny = 2\nz = x + y",
            new_code="z = add(1, 2)",
            risk_score=0.3,
        )
        operation2 = RefactoringOperation(
            operation_type="extract_constant",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract constant",
            old_code="x = 1",
            new_code="MAGIC_NUMBER = 1",
            risk_score=0.1,
        )

        hash1 = self.fingerprinter.fingerprint_refactoring(operation1)
        hash2 = self.fingerprinter.fingerprint_refactoring(operation2)

        # Different operations should produce different hashes
        assert hash1 != hash2

    def test_fingerprint_refactoring_same_old_code(self):
        """Test that same old_code with different operation_type produces different hashes."""
        old_code = "x = 1\ny = 2\nz = x + y"
        operation1 = RefactoringOperation(
            operation_type="extract_method",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract method",
            old_code=old_code,
            new_code="z = add(1, 2)",
            risk_score=0.3,
        )
        operation2 = RefactoringOperation(
            operation_type="extract_constant",
            file_path=Path("test.py"),
            line_number=10,
            description="Extract constant",
            old_code=old_code,
            new_code="MAGIC_NUMBER = 1",
            risk_score=0.1,
        )

        hash1 = self.fingerprinter.fingerprint_refactoring(operation1)
        hash2 = self.fingerprinter.fingerprint_refactoring(operation2)

        # Same old_code but different operation_type should produce different hashes
        assert hash1 != hash2

    def test_normalize_code_removes_comments(self):
        """Test that code normalization removes comments."""
        code = "def add(a, b):\n    # This is a comment\n    return a + b"
        normalized = self.fingerprinter._normalize_code(code)

        assert "#" not in normalized
        assert "comment" not in normalized.lower()

    def test_normalize_code_removes_docstrings(self):
        """Test that code normalization removes docstrings."""
        code = 'def add(a, b):\n    """Add two numbers."""\n    return a + b'
        normalized = self.fingerprinter._normalize_code(code)

        assert '"""' not in normalized
        assert "Add two numbers" not in normalized

    def test_normalize_code_empty_code(self):
        """Test normalizing empty code."""
        normalized = self.fingerprinter._normalize_code("")
        assert normalized == ""

    def test_normalize_code_whitespace_only(self):
        """Test normalizing whitespace-only code."""
        normalized = self.fingerprinter._normalize_code("   \n\n  \n")
        assert normalized == "" or normalized.strip() == ""

    def test_extract_ast_pattern_function(self):
        """Test AST pattern extraction for functions."""
        code = "def hello():\n    print('world')"
        pattern = self.fingerprinter._extract_ast_pattern(code)

        assert "FUNC" in pattern
        assert "CALL" in pattern

    def test_extract_ast_pattern_class(self):
        """Test AST pattern extraction for classes."""
        code = "class MyClass:\n    def method(self):\n        pass"
        pattern = self.fingerprinter._extract_ast_pattern(code)

        assert "CLASS" in pattern
        assert "FUNC" in pattern  # method function

    def test_extract_ast_pattern_control_flow(self):
        """Test AST pattern extraction for control flow."""
        code = "if x > 0:\n    for i in range(10):\n        while True:\n            break"
        pattern = self.fingerprinter._extract_ast_pattern(code)

        assert "IF" in pattern
        assert "FOR" in pattern
        assert "WHILE" in pattern

    def test_extract_ast_pattern_invalid_syntax(self):
        """Test AST pattern extraction for invalid syntax."""
        code = "def invalid syntax here!!!"
        pattern = self.fingerprinter._extract_ast_pattern(code)

        # Should return empty pattern for invalid syntax
        assert pattern == ""
