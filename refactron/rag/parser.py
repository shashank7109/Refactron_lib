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

        # Initialize Python language - handle different tree-sitter API versions
        lang = tspython.language()

        # In some versions, tspython.language() already returns a Language object
        if isinstance(lang, Language):
            PY_LANGUAGE = lang
        else:
            # Try newer API first (single argument)
            try:
                PY_LANGUAGE = Language(lang)
            except TypeError:
                # Try older API (needs name)
                try:
                    PY_LANGUAGE = Language(lang, "python")
                except TypeError:
                    # Try using the path to the compiled library (for very old or CI bindings)
                    try:
                        import os
                        import platform

                        pkg_dir = os.path.dirname(tspython.__file__)

                        # Find the correct shared library extension
                        system = platform.system()
                        if system == "Windows":
                            ext = ".dll"
                        elif system == "Darwin":
                            ext = ".dylib"
                        else:
                            ext = ".so"

                        # Look for common names of the compiled language file
                        lib_path = None
                        for fname in os.listdir(pkg_dir):
                            if fname.endswith(ext):
                                lib_path = os.path.join(pkg_dir, fname)
                                break

                        if lib_path:
                            PY_LANGUAGE = Language(lib_path, "python")
                        else:
                            # Last resort: try as keyword or whatever lang is
                            PY_LANGUAGE = Language(lang, name="python")
                    except Exception:
                        # Absolute last resort
                        PY_LANGUAGE = Language(lang, name="python")

        self.parser = Parser(PY_LANGUAGE)

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
