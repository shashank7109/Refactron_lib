"""Pattern fingerprinting for code pattern identification."""

import ast
import hashlib
import re

from refactron.core.models import CodeIssue, RefactoringOperation


class PatternFingerprinter:
    """Generates fingerprints for code patterns using AST-based hashing."""

    def __init__(self) -> None:
        """Initialize the pattern fingerprinter."""
        self._hash_algo = hashlib.sha256

    def fingerprint_code(self, code_snippet: str) -> str:
        """
        Generate hash fingerprint for a code snippet.

        Args:
            code_snippet: Source code to fingerprint

        Returns:
            SHA256 hash of the normalized code pattern
        """
        if not code_snippet:
            return self._hash_algo(b"").hexdigest()

        # Optimize: Parse AST once, extract both normalized code and pattern
        # Anonymize identifiers to allow structural generalization
        anonymized = self._anonymize_code(code_snippet)
        ast_pattern = self._extract_ast_pattern(code_snippet)
        combined = f"{anonymized}\n{ast_pattern}".encode("utf-8")
        return self._hash_algo(combined).hexdigest()

    def fingerprint_issue_context(
        self, issue: CodeIssue, source_code: str, context_lines: int = 3
    ) -> str:
        """
        Generate fingerprint for issue context.

        Args:
            issue: CodeIssue to fingerprint
            source_code: Full source code of the file
            context_lines: Number of lines before/after to include (default: 3)

        Returns:
            SHA256 hash of the normalized issue context pattern
        """
        lines = source_code.split("\n")
        start_line = max(0, issue.line_number - context_lines - 1)
        end_line = min(len(lines), issue.line_number + context_lines)

        context_code = "\n".join(lines[start_line:end_line])

        # Include issue category and rule_id in the fingerprint
        issue_metadata = f"{issue.category.value}:{issue.rule_id or 'none'}"
        combined = f"{self._normalize_code(context_code)}\n{issue_metadata}".encode("utf-8")

        return self._hash_algo(combined).hexdigest()

    def fingerprint_refactoring(self, operation: RefactoringOperation) -> str:
        """
        Generate fingerprint for refactoring operation.

        Args:
            operation: RefactoringOperation to fingerprint

        Returns:
            SHA256 hash of the normalized refactoring pattern
        """
        # Combine old_code pattern + operation_type for unique identification
        # Use anonymization to group similar structural refactorings
        anonymized_old = self._anonymize_code(operation.old_code)
        operation_key = f"{operation.operation_type}:{anonymized_old}"
        combined = operation_key.encode("utf-8")

        return self._hash_algo(combined).hexdigest()

    def _normalize_code(self, code: str) -> str:
        """
        Normalize code by removing comments, normalizing whitespace.

        Optimized to minimize string operations and passes.

        Args:
            code: Source code to normalize

        Returns:
            Normalized code string
        """
        if not code:
            return ""

        # Optimize: Normalize line endings first (single pass)
        normalized = code.replace("\r\n", "\n").replace("\r", "\n")

        # Optimize: Remove multi-line docstrings before line-by-line processing
        normalized = re.sub(r'""".*?"""', "", normalized, flags=re.DOTALL)
        normalized = re.sub(r"'''.*?'''", "", normalized, flags=re.DOTALL)

        # Process lines in single pass
        lines = normalized.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            # Remove # comments (but preserve if inside string)
            comment_pos = line.find("#")
            if comment_pos >= 0:
                # Check if # is inside a string (simple heuristic)
                before_hash = line[:comment_pos]
                quote_count = before_hash.count('"') + before_hash.count("'")
                if quote_count % 2 == 0:  # Even means not in string
                    line = line[:comment_pos].rstrip()
                else:
                    line = line.rstrip()

            # Normalize whitespace: multiple spaces to single space
            line = re.sub(r" +", " ", line.rstrip())
            cleaned_lines.append(line)

        # Remove empty lines at start/end, but preserve internal structure
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    def _extract_ast_pattern(self, code: str) -> str:
        """
        Extract structural pattern from AST.

        Args:
            code: Source code to analyze

        Returns:
            String representation of AST structure
        """
        try:
            tree = ast.parse(code)
            pattern_parts = []

            for node in ast.walk(tree):
                node_type = type(node).__name__

                # Extract key structural elements
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    pattern_parts.append("FUNC")
                elif isinstance(node, ast.ClassDef):
                    pattern_parts.append("CLASS")
                elif isinstance(node, ast.If):
                    pattern_parts.append("IF")
                elif isinstance(node, ast.For):
                    pattern_parts.append("FOR")
                elif isinstance(node, ast.While):
                    pattern_parts.append("WHILE")
                elif isinstance(node, ast.Try):
                    pattern_parts.append("TRY")
                elif isinstance(node, ast.With):
                    pattern_parts.append("WITH")
                elif isinstance(node, ast.Assign):
                    pattern_parts.append("ASSIGN")
                elif isinstance(node, ast.Return):
                    pattern_parts.append("RETURN")
                elif isinstance(node, ast.Call):
                    pattern_parts.append("CALL")
                elif isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
                    pattern_parts.append(f"COLLECTION:{node_type}")

            return "|".join(sorted(set(pattern_parts)))  # Sorted for consistency

        except (SyntaxError, ValueError):
            # If code is invalid, return empty pattern
            return ""

    def _anonymize_code(self, code: str) -> str:
        """
        Normalize and anonymize identifiers to help patterns generalize.

        Replaces specific function/variable names with generic tokens.
        """
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    node.id = "VAR"
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    node.name = "FUNC"
                elif isinstance(node, ast.ClassDef):
                    node.name = "CLASS"
                elif isinstance(node, ast.Attribute):
                    node.attr = "ATTR"
                elif isinstance(node, ast.arg):
                    node.arg = "ARG"
                elif isinstance(node, ast.Constant):
                    # Anonymize all literals (numbers, strings, etc)
                    node.value = "CONST"
                elif isinstance(node, ast.JoinedStr):
                    # Anonymize f-strings by replacing constant parts
                    for i, value in enumerate(node.values):
                        if isinstance(value, ast.Constant):
                            node.values[i] = ast.Constant(value="STR")

            # Using ast.unparse if available (Python 3.9+)
            if hasattr(ast, "unparse"):
                return self._normalize_code(ast.unparse(tree))

            # Fallback to standard normalization if unparse is not available
            return self._normalize_code(code)

        except (SyntaxError, ValueError):
            # If AST parsing fails, fallback to basic normalization
            return self._normalize_code(code)
