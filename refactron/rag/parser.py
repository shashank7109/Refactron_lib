"""Code parser using tree-sitter for AST-aware code analysis."""

from __future__ import annotations

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

        PY_LANGUAGE = self._build_language()
        self.parser = self._build_parser(PY_LANGUAGE)

    @staticmethod
    def _tree_sitter_minor_version() -> int:
        """Return the installed tree-sitter minor version (e.g. 20, 21, 22).

        Uses ``importlib.metadata`` which is part of the stdlib since Python 3.8
        and is always accurate for the active virtual environment.
        """
        import importlib.metadata

        try:
            raw = importlib.metadata.version("tree-sitter")
            # raw is e.g. "0.21.0" or "0.22.6" or "0.25.2"
            parts = raw.split(".")
            return int(parts[1]) if len(parts) >= 2 else 0
        except Exception:
            return 0

    @staticmethod
    def _build_language() -> "Language":
        """Construct a tree-sitter Language object for the installed API version.

        tree-sitter changed its public API across three version ranges:

        - 0.20.x  ``Language(path_to_so, "python")``  — shared-library path + name
        - 0.21.x  ``Language(capsule, "python")``      — capsule + name
        - 0.22+   ``Language(capsule)``                — capsule only (name dropped)

        We read the actual installed version via ``importlib.metadata`` and
        dispatch directly to the correct form, avoiding any silent-failure risk
        from blind try/except probing.
        """
        import os
        import platform

        lang = tspython.language()

        # Already a fully constructed Language object (rare binding variant)
        if isinstance(lang, Language):
            return lang

        minor = CodeParser._tree_sitter_minor_version()

        # ── 0.22 and newer ──────────────────────────────────────────────────
        if minor >= 22:
            return Language(lang)

        # ── 0.21.x ──────────────────────────────────────────────────────────
        if minor == 21:
            try:
                return Language(lang, "python")
            except TypeError:
                # Some 0.21 builds already dropped the name argument
                return Language(lang)

        # ── 0.20.x ──────────────────────────────────────────────────────────
        # Language() requires the path to the compiled shared library
        pkg_dir = os.path.dirname(tspython.__file__)
        system = platform.system()
        ext = ".dll" if system == "Windows" else (".dylib" if system == "Darwin" else ".so")

        for fname in sorted(os.listdir(pkg_dir)):
            if fname.endswith(ext):
                lib_path = os.path.join(pkg_dir, fname)
                try:
                    return Language(lib_path, "python")
                except (TypeError, OSError):
                    continue

        # Unknown / future version — try both forms as a last resort
        try:
            return Language(lang)
        except TypeError:
            pass
        try:
            return Language(lang, "python")
        except TypeError:
            pass

        raise RuntimeError(
            f"Could not construct a tree-sitter Language for Python "
            f"(tree-sitter minor version: {minor}). "
            "Try: pip install --upgrade tree-sitter tree-sitter-python"
        )

    @staticmethod
    def _build_parser(language: "Language") -> "Parser":
        """Construct a tree-sitter Parser compatible with the installed API version.

        - 0.20.x  ``Parser()`` then ``parser.set_language(language)``
        - 0.21+   ``Parser(language)``
        """
        minor = CodeParser._tree_sitter_minor_version()

        if minor >= 21:
            try:
                parser = Parser(language)
                if parser.parse(b"x = 1") is not None:
                    return parser
            except TypeError:
                pass  # fall through to set_language path

        # 0.20.x and any fallback path
        parser = Parser()
        parser.set_language(language)
        if parser.parse(b"x = 1") is None:
            raise RuntimeError(
                "tree-sitter Parser could not be initialised. "
                "Try: pip install --upgrade tree-sitter tree-sitter-python"
            )
        return parser

    def parse_file(self, file_path: Path) -> ParsedFile:
        """Parse a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            ParsedFile object containing all parsed elements
        """
        with open(file_path, "rb") as f:
            source_code = f.read()

        tree = self.parser.parse(source_code)
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
