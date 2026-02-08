"""Code chunking strategies for RAG indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from refactron.rag.parser import CodeParser, ParsedClass, ParsedFile, ParsedFunction


@dataclass
class CodeChunk:
    """Represents a semantic chunk of code."""

    content: str
    chunk_type: str  # "function", "class", "module"
    file_path: str
    line_range: Tuple[int, int]
    name: str
    dependencies: List[str]
    metadata: Dict[str, Any]


class CodeChunker:
    """Chunks parsed code into semantic units for embedding."""

    def __init__(self, parser: CodeParser):
        """Initialize the chunker.

        Args:
            parser: CodeParser instance for parsing files
        """
        self.parser = parser

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """Chunk a file into semantic units.

        Args:
            file_path: Path to the Python file

        Returns:
            List of code chunks
        """
        # Parse the file first
        parsed_file = self.parser.parse_file(file_path)

        chunks = []

        # Add module-level chunk if there's a docstring or imports
        if parsed_file.module_docstring or parsed_file.imports:
            chunks.append(self._create_module_chunk(parsed_file))

        # Add function chunks
        for func in parsed_file.functions:
            chunks.append(self._create_function_chunk(func, parsed_file.file_path))

        # Add class chunks
        for cls in parsed_file.classes:
            chunks.extend(self._create_class_chunks(cls, parsed_file.file_path))

        return chunks

    def _create_module_chunk(self, parsed_file: ParsedFile) -> CodeChunk:
        """Create a chunk for module-level information."""
        content_parts = []

        if parsed_file.module_docstring:
            content_parts.append(f'"""{parsed_file.module_docstring}"""')

        if parsed_file.imports:
            content_parts.append("\n".join(parsed_file.imports))

        # Add context header
        header = f"File: {parsed_file.file_path}, Type: Module"
        content = header + "\n" + "-" * len(header) + "\n\n" + "\n\n".join(content_parts)

        return CodeChunk(
            content=content,
            chunk_type="module",
            file_path=parsed_file.file_path,
            line_range=(1, len(parsed_file.imports) + 1),
            name="module",
            dependencies=parsed_file.imports,
            metadata={
                "docstring": parsed_file.module_docstring,
                "num_imports": len(parsed_file.imports),
            },
        )

    def _create_function_chunk(self, func: ParsedFunction, file_path: str) -> CodeChunk:
        """Create a chunk for a function."""
        # Build content with context
        content_parts = []

        if func.docstring:
            content_parts.append(f'"""{func.docstring}"""')

        content_parts.append(func.body)

        # Add context header
        header = f"File: {file_path}, Function: {func.name}"
        content = header + "\n" + "-" * len(header) + "\n\n" + "\n".join(content_parts)

        return CodeChunk(
            content=content,
            chunk_type="function",
            file_path=file_path,
            line_range=func.line_range,
            name=func.name,
            dependencies=[],  # Could extract from body if needed
            metadata={
                "docstring": func.docstring,
                "params": func.params,
                "num_params": len(func.params),
            },
        )

    def _create_class_chunks(self, cls: ParsedClass, file_path: str) -> List[CodeChunk]:
        """Create chunks for a class and its methods."""
        chunks = []

        # Create class overview chunk
        class_content_parts = []

        if cls.docstring:
            class_content_parts.append(f'"""{cls.docstring}"""')

        # Include class signature and docstring
        class_signature = cls.body.split("\n")[0]  # First line
        class_content_parts.append(class_signature)

        # Add context header
        header = f"File: {file_path}, Class: {cls.name}"
        content = header + "\n" + "-" * len(header) + "\n\n" + "\n".join(class_content_parts)

        class_chunk = CodeChunk(
            content=content,
            chunk_type="class",
            file_path=file_path,
            line_range=cls.line_range,
            name=cls.name,
            dependencies=[],
            metadata={
                "docstring": cls.docstring,
                "num_methods": len(cls.methods),
                "methods": [m.name for m in cls.methods],
            },
        )
        chunks.append(class_chunk)

        # Create chunks for each method
        for method in cls.methods:
            # Add context header specifically for methods
            header = f"File: {file_path}, Class: {cls.name}, Method: {method.name}"
            content_parts = []
            if method.docstring:
                content_parts.append(f'"""{method.docstring}"""')
            content_parts.append(method.body)

            content = header + "\n" + "-" * len(header) + "\n\n" + "\n".join(content_parts)

            method_chunk = CodeChunk(
                content=content,
                chunk_type="method",
                file_path=file_path,
                line_range=method.line_range,
                name=method.name,
                dependencies=[],
                metadata={
                    "docstring": method.docstring,
                    "params": method.params,
                    "num_params": len(method.params),
                    "class_name": cls.name,
                },
            )
            chunks.append(method_chunk)

        return chunks
