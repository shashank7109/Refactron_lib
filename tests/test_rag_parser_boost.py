"""Tests for rag/parser.py – CodeParser and data classes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestParsedDataclasses:
    def test_parsed_function_fields(self):
        from refactron.rag.parser import ParsedFunction

        f = ParsedFunction(
            name="foo", body="def foo(): pass", docstring=None, line_range=(1, 1), params=["x"]
        )
        assert f.name == "foo"

    def test_parsed_class_fields(self):
        from refactron.rag.parser import ParsedClass, ParsedFunction

        c = ParsedClass(
            name="MyClass",
            body="class MyClass: pass",
            docstring="A class",
            line_range=(1, 5),
            methods=[],
        )
        assert c.name == "MyClass"

    def test_parsed_file_fields(self):
        from refactron.rag.parser import ParsedFile

        f = ParsedFile(
            file_path="a.py",
            imports=["import os"],
            functions=[],
            classes=[],
            module_docstring="Module docs",
        )
        assert f.file_path == "a.py"


class TestCodeParserUnavailable:
    def test_raises_when_tree_sitter_unavailable(self):
        with patch("refactron.rag.parser.TREE_SITTER_AVAILABLE", False):
            from importlib import reload

            import refactron.rag.parser as parser_mod

            with patch.object(parser_mod, "TREE_SITTER_AVAILABLE", False):
                with pytest.raises(RuntimeError, match="tree-sitter"):
                    parser_mod.CodeParser()


class TestCodeParserAvailable:
    @pytest.fixture()
    def parser(self):
        pytest.importorskip("tree_sitter")
        pytest.importorskip("tree_sitter_python")
        from refactron.rag.parser import CodeParser

        return CodeParser()

    def test_parse_simple_function(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello(x, y):\n    '''Say hello.'''\n    return x + y\n")
        result = parser.parse_file(f)
        assert result.file_path == str(f)
        func_names = [fn.name for fn in result.functions]
        assert "hello" in func_names

    def test_parse_function_params(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def greet(name, greeting='Hi'):\n    pass\n")
        result = parser.parse_file(f)
        fn = result.functions[0]
        assert "name" in fn.params

    def test_parse_class(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class Foo:\n    def bar(self):\n        pass\n")
        result = parser.parse_file(f)
        assert len(result.classes) == 1
        assert result.classes[0].name == "Foo"
        assert len(result.classes[0].methods) == 1

    def test_parse_imports(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("import os\nfrom pathlib import Path\n\nx = 1\n")
        result = parser.parse_file(f)
        assert len(result.imports) == 2

    def test_parse_module_docstring(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('"""Module docstring."""\n\nx = 1\n')
        result = parser.parse_file(f)
        assert result.module_docstring is not None

    def test_parse_function_docstring(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('def foo():\n    """Foo docs."""\n    pass\n')
        result = parser.parse_file(f)
        assert result.functions[0].docstring is not None

    def test_parse_class_docstring(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('class Bar:\n    """Bar docs."""\n    pass\n')
        result = parser.parse_file(f)
        assert result.classes[0].docstring is not None

    def test_parse_empty_file(self, parser, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = parser.parse_file(f)
        assert result.functions == [] and result.classes == []

    def test_parse_no_docstring(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def no_doc():\n    return 1\n")
        result = parser.parse_file(f)
        assert result.functions[0].docstring is None

    def test_parse_class_no_docstring(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class Bare:\n    pass\n")
        result = parser.parse_file(f)
        assert result.classes[0].docstring is None

    def test_line_range(self, parser, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    x = 1\n    return x\n")
        result = parser.parse_file(f)
        assert result.functions[0].line_range[0] == 1

    def test_parse_nested_class_methods(self, parser, tmp_path):
        f = tmp_path / "test.py"
        code = "class A:\n    def m1(self): pass\n    def m2(self): pass\n"
        f.write_text(code)
        result = parser.parse_file(f)
        assert len(result.classes[0].methods) == 2


class TestCodeParserTreeSitterVersion:
    def test_minor_version_fallback(self):
        with patch("refactron.rag.parser.TREE_SITTER_AVAILABLE", True):
            import tree_sitter

            # Just test it doesn't raise
            from refactron.rag.parser import CodeParser

            v = CodeParser._tree_sitter_minor_version()
            assert isinstance(v, int)

    def test_parser_works_valid(self):
        pytest.importorskip("tree_sitter")
        from refactron.rag.parser import CodeParser

        mock_parser = MagicMock()
        mock_parser.parse.return_value = MagicMock()  # Non-None
        assert CodeParser._parser_works(mock_parser) is True

    def test_parser_works_returns_none(self):
        pytest.importorskip("tree_sitter")
        from refactron.rag.parser import CodeParser

        mock_parser = MagicMock()
        mock_parser.parse.return_value = None
        assert CodeParser._parser_works(mock_parser) is False

    def test_parser_works_raises(self):
        pytest.importorskip("tree_sitter")
        from refactron.rag.parser import CodeParser

        mock_parser = MagicMock()
        mock_parser.parse.side_effect = Exception("parse error")
        assert CodeParser._parser_works(mock_parser) is False
