"""Code parser using tree-sitter for AST-aware code analysis."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Node, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class ParsedFunction:
    """Represents a parsed function."""

    name: str
    body: str
    docstring: Optional[str]
    line_range: Tuple[int, int]
    params: List[str]


@dataclass
class ParsedClass:
    """Represents a parsed class."""

    name: str
    body: str
    docstring: Optional[str]
    line_range: Tuple[int, int]
    methods: List[ParsedFunction]


@dataclass
class ParsedFile:
    """Represents a parsed Python file."""

    file_path: str
    imports: List[str]
    functions: List[ParsedFunction]
    classes: List[ParsedClass]
    module_docstring: Optional[str]


class CodeParser:
    """AST-aware code parser using tree-sitter."""

    def __init__(self) -> None:
        """Initialize the parser."""
        if not TREE_SITTER_AVAILABLE:
            raise RuntimeError(
                "tree-sitter is not available. "
                "Install with: pip install tree-sitter tree-sitter-python"
            )

        language = CodeParser._build_language()
        self.parser = CodeParser._build_parser(language)

    @staticmethod
    def _tree_sitter_minor_version() -> int:
        """Return the minor version of the installed tree-sitter package."""
        import tree_sitter

        version = getattr(tree_sitter, "__version__", "0.20.0")
        parts = str(version).split(".")
        if len(parts) < 2:
            return 20
        try:
            return int(parts[1])
        except ValueError:
            return 20

    @staticmethod
    def _build_language() -> "Language":
        """Build a ``Language`` for Python across tree-sitter / binding versions."""
        lang = tspython.language()

        if isinstance(lang, Language):
            return lang

        minor = CodeParser._tree_sitter_minor_version()

        # 0.21.x: tspython.language() may return a raw int (C pointer).
        if isinstance(lang, int):
            try:
                return Language(lang, "python")
            except TypeError:
                pass

        # 0.22+: PyCapsule — single-argument Language constructor.
        if minor >= 22:
            try:
                return Language(lang)
            except TypeError:
                pass

        # 0.20.x: Language needs path to the compiled grammar shared library.
        pkg_dir = os.path.dirname(tspython.__file__)
        system = platform.system()
        ext = ".dll" if system == "Windows" else (".dylib" if system == "Darwin" else ".so")

        for fname in sorted(os.listdir(pkg_dir)):
            if fname.endswith(ext):
                lib_path = os.path.join(pkg_dir, fname)
                try:
                    return Language(lib_path, "python")
                except (TypeError, OSError, ValueError):
                    continue

        # Fallbacks for unusual binding combinations.
        try:
            return Language(lang, "python")
        except TypeError:
            pass
        try:
            return Language(lang)
        except TypeError:
            pass

        raise RuntimeError(
            "Could not construct tree-sitter Language for Python. "
            "Try: pip install --upgrade tree-sitter tree-sitter-python"
        )

    @staticmethod
    def _parser_works(parser: "Parser") -> bool:
        """Return True if the parser can successfully parse trivial Python code.

        On tree-sitter 0.21.x, ``parse()`` raises ``ValueError`` for a parser
        that has no language set; on 0.22+ it returns ``None``.  We handle both.
        """
        try:
            result = parser.parse(b"x = 1")
            return result is not None
        except (ValueError, RuntimeError):
            return False

    @staticmethod
    def _build_parser(language: "Language") -> "Parser":
        """Construct a tree-sitter Parser compatible with the installed API version.

        API history:
        - 0.20.x  ``Parser()`` then ``parser.set_language(language)``
        - 0.21.x  ``Parser()`` then ``parser.set_language(language)``
          (``Parser(language)`` raises ``TypeError`` — no positional args)
        - 0.22+   ``Parser(language)`` — language passed to constructor directly

        On 0.21.x ``parser.parse()`` raises ``ValueError`` instead of returning
        ``None`` when called on a parser with no language set, so both behaviours
        are handled in ``_parser_works()``.
        """
        minor = CodeParser._tree_sitter_minor_version()

        # 0.22+ accepts Language in the constructor
        if minor >= 22:
            try:
                parser = Parser(language)
                if CodeParser._parser_works(parser):
                    return parser
            except TypeError:
                pass  # unexpected; fall through to set_language path

        # 0.20.x and 0.21.x: no-arg constructor + set_language()
        try:
            parser = Parser()
            parser.set_language(language)
            if CodeParser._parser_works(parser):
                return parser
        except (TypeError, AttributeError):
            pass

        raise RuntimeError(
            f"tree-sitter Parser could not be initialised "
            f"(tree-sitter minor version: {CodeParser._tree_sitter_minor_version()}). "
            "Try: pip install --upgrade tree-sitter tree-sitter-python"
        )

    def parse_file(self, file_path: Path) -> ParsedFile:
        """Parse a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            ParsedFile object containing all parsed elements
        """
        with open(file_path, "rb") as f:
            source_code = f.read()

        # tree-sitter 0.22+ returns None on failure; 0.21.x raises ValueError.
        # We normalise both to a single ValueError with an informative message.
        try:
            tree = self.parser.parse(source_code)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Parsing failed for file {file_path}: {exc}") from exc

        if tree is None:
            raise ValueError(
                f"Parsing failed for file {file_path}. "
                "The file may contain syntax errors or use unsupported Python features."
            )
        root = tree.root_node

        # Extract module docstring
        module_docstring = self._extract_module_docstring(root, source_code)

        # Extract imports
        imports = self._extract_imports(root, source_code)

        # Extract functions
        functions = self._extract_functions(root, source_code)

        # Extract classes
        classes = self._extract_classes(root, source_code)

        return ParsedFile(
            file_path=str(file_path),
            imports=imports,
            functions=functions,
            classes=classes,
            module_docstring=module_docstring,
        )

    def _extract_module_docstring(self, root: Node, source: bytes) -> Optional[str]:
        """Extract module-level docstring."""
        for child in root.children:
            if child.type == "expression_statement":
                string_node = child.children[0] if child.children else None
                if string_node and string_node.type == "string":
                    return (
                        source[string_node.start_byte : string_node.end_byte]
                        .decode("utf-8")
                        .strip("\"'")
                    )
        return None

    def _extract_imports(self, root: Node, source: bytes) -> List[str]:
        """Extract import statements."""
        imports = []
        for node in root.children:
            if node.type in ("import_statement", "import_from_statement"):
                import_text = source[node.start_byte : node.end_byte].decode("utf-8")
                imports.append(import_text)
        return imports

    def _extract_functions(self, root: Node, source: bytes) -> List[ParsedFunction]:
        """Extract function definitions."""
        functions = []
        for node in root.children:
            if node.type == "function_definition":
                func = self._parse_function(node, source)
                if func:
                    functions.append(func)
        return functions

    def _extract_classes(self, root: Node, source: bytes) -> List[ParsedClass]:
        """Extract class definitions."""
        classes = []
        for node in root.children:
            if node.type == "class_definition":
                cls = self._parse_class(node, source)
                if cls:
                    classes.append(cls)
        return classes

    def _parse_function(self, node: Node, source: bytes) -> Optional[ParsedFunction]:
        """Parse a function node."""
        # Get function name
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Get function body
        body = source[node.start_byte : node.end_byte].decode("utf-8")

        # Get docstring
        docstring = self._extract_function_docstring(node, source)

        # Get line range
        line_range = (node.start_point[0] + 1, node.end_point[0] + 1)

        # Get parameters
        params = self._extract_parameters(node, source)

        return ParsedFunction(
            name=name,
            body=body,
            docstring=docstring,
            line_range=line_range,
            params=params,
        )

    def _parse_class(self, node: Node, source: bytes) -> Optional[ParsedClass]:
        """Parse a class node."""
        # Get class name
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Get class body
        body = source[node.start_byte : node.end_byte].decode("utf-8")

        # Get docstring
        docstring = self._extract_class_docstring(node, source)

        # Get line range
        line_range = (node.start_point[0] + 1, node.end_point[0] + 1)

        # Extract methods from class body
        methods = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    method = self._parse_function(child, source)
                    if method:
                        methods.append(method)

        return ParsedClass(
            name=name,
            body=body,
            docstring=docstring,
            line_range=line_range,
            methods=methods,
        )

    def _extract_function_docstring(self, node: Node, source: bytes) -> Optional[str]:
        """Extract function docstring."""
        body_node = node.child_by_field_name("body")
        if not body_node:
            return None

        for child in body_node.children:
            if child.type == "expression_statement":
                string_node = child.children[0] if child.children else None
                if string_node and string_node.type == "string":
                    return (
                        source[string_node.start_byte : string_node.end_byte]
                        .decode("utf-8")
                        .strip("\"'")
                    )
        return None

    def _extract_class_docstring(self, node: Node, source: bytes) -> Optional[str]:
        """Extract class docstring."""
        body_node = node.child_by_field_name("body")
        if not body_node:
            return None

        for child in body_node.children:
            if child.type == "expression_statement":
                string_node = child.children[0] if child.children else None
                if string_node and string_node.type == "string":
                    return (
                        source[string_node.start_byte : string_node.end_byte]
                        .decode("utf-8")
                        .strip("\"'")
                    )
        return None

    def _extract_parameters(self, node: Node, source: bytes) -> List[str]:
        """Extract function parameters."""
        params: List[str] = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "identifier":
                param_name = source[child.start_byte : child.end_byte].decode("utf-8")
                params.append(param_name)

        return params
