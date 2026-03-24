"""Tests for the RAG parser module."""

import tempfile
from pathlib import Path

import pytest

from refactron.rag.parser import (
    TREE_SITTER_AVAILABLE,
    CodeParser,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
)


def _tree_sitter_usable() -> bool:
    if not TREE_SITTER_AVAILABLE:
        return False
    try:
        CodeParser()
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _tree_sitter_usable(),
    reason="tree-sitter is not available or cannot be initialised in this environment",
)


class TestCodeParser:
    """Test cases for CodeParser."""

    @pytest.fixture
    def parser(self):
        """Create a CodeParser instance."""
        return CodeParser()

    @pytest.fixture
    def temp_python_file(self):
        """Create a temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            content = '''"""Module docstring for testing."""

import os
import sys
from pathlib import Path

def simple_function(x, y):
    """Add two numbers."""
    return x + y

def another_function():
    """Function without params."""
    pass

class TestClass:
    """A test class."""

    def method_one(self):
        """First method."""
        return 1

    def method_two(self, param):
        """Second method."""
        return param * 2
'''
            f.write(content)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        temp_path.unlink()

    def test_parser_initialization(self, parser):
        """Test that parser initializes correctly."""
        assert parser is not None
        assert parser.parser is not None

    def test_parse_file_basic(self, parser, temp_python_file):
        """Test parsing a basic Python file."""
        parsed = parser.parse_file(temp_python_file)

        assert isinstance(parsed, ParsedFile)
        assert parsed.file_path == str(temp_python_file)
        assert parsed.module_docstring == "Module docstring for testing."

    def test_extract_imports(self, parser, temp_python_file):
        """Test that imports are extracted correctly."""
        parsed = parser.parse_file(temp_python_file)

        assert len(parsed.imports) == 3
        assert "import os" in parsed.imports
        assert "import sys" in parsed.imports
        assert any("pathlib" in imp for imp in parsed.imports)

    def test_extract_functions(self, parser, temp_python_file):
        """Test that functions are extracted correctly."""
        parsed = parser.parse_file(temp_python_file)

        assert len(parsed.functions) == 2

        # Check first function
        func1 = parsed.functions[0]
        assert isinstance(func1, ParsedFunction)
        assert func1.name == "simple_function"
        assert func1.docstring == "Add two numbers."
        assert len(func1.params) >= 2  # Should have x and y

        # Check second function
        func2 = parsed.functions[1]
        assert func2.name == "another_function"

    def test_extract_classes(self, parser, temp_python_file):
        """Test that classes are extracted correctly."""
        parsed = parser.parse_file(temp_python_file)

        assert len(parsed.classes) == 1

        # Check class
        cls = parsed.classes[0]
        assert isinstance(cls, ParsedClass)
        assert cls.name == "TestClass"
        assert cls.docstring == "A test class."

        # Check methods
        assert len(cls.methods) == 2
        assert cls.methods[0].name == "method_one"
        assert cls.methods[1].name == "method_two"

    def test_parse_invalid_file(self, parser):
        """Test parsing a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.py"))

    def test_parse_empty_file(self, parser):
        """Test parsing an empty Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            parsed = parser.parse_file(temp_path)
            assert parsed.module_docstring is None
            assert len(parsed.imports) == 0
            assert len(parsed.functions) == 0
            assert len(parsed.classes) == 0
        finally:
            temp_path.unlink()

    def test_function_line_ranges(self, parser, temp_python_file):
        """Test that line ranges are captured correctly."""
        parsed = parser.parse_file(temp_python_file)

        for func in parsed.functions:
            assert func.line_range[0] > 0
            assert func.line_range[1] >= func.line_range[0]

    def test_class_methods_have_correct_metadata(self, parser, temp_python_file):
        """Test that class methods preserve metadata."""
        parsed = parser.parse_file(temp_python_file)

        test_class = parsed.classes[0]
        for method in test_class.methods:
            assert method.name in ["method_one", "method_two"]
            assert method.docstring is not None
            assert len(method.body) > 0
