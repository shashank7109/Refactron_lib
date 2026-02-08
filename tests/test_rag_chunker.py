"""Tests for the RAG chunker module."""

import tempfile
from pathlib import Path

import pytest

from refactron.rag.chunker import CodeChunker, CodeChunk
from refactron.rag.parser import CodeParser


class TestCodeChunker:
    """Test cases for CodeChunker."""

    @pytest.fixture
    def parser(self):
        """Create a CodeParser instance."""
        return CodeParser()

    @pytest.fixture
    def chunker(self, parser):
        """Create a CodeChunker instance."""
        return CodeChunker(parser)

    @pytest.fixture
    def temp_python_file(self):
        """Create a temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            content = '''"""Module level docstring."""

import os
import sys

def test_function(x):
    """Test function docstring."""
    return x * 2

def another_function():
    """Another function."""
    pass

class TestClass:
    """Test class docstring."""
    
    def method_one(self):
        """Method docstring."""
        return 1
'''
            f.write(content)
            temp_path = Path(f.name)

        yield temp_path

        # Cleanup
        temp_path.unlink()

    def test_chunker_initialization(self, parser, chunker):
        """Test that chunker initializes with parser."""
        assert chunker.parser is not None
        assert chunker.parser == parser

    def test_chunk_file_basic(self, chunker, temp_python_file):
        """Test basic file chunking."""
        chunks = chunker.chunk_file(temp_python_file)

        assert len(chunks) > 0
        assert all(isinstance(chunk, CodeChunk) for chunk in chunks)

    def test_module_chunk_created(self, chunker, temp_python_file):
        """Test that module chunk is created when there are imports/docstring."""
        chunks = chunker.chunk_file(temp_python_file)

        module_chunks = [c for c in chunks if c.chunk_type == "module"]
        assert len(module_chunks) == 1

        module_chunk = module_chunks[0]
        assert "Module level docstring" in module_chunk.content
        assert "import os" in module_chunk.content

    def test_function_chunks_created(self, chunker, temp_python_file):
        """Test that function chunks are created correctly."""
        chunks = chunker.chunk_file(temp_python_file)

        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(function_chunks) == 2

        # Check first function chunk
        func_names = [c.name for c in function_chunks]
        assert "test_function" in func_names
        assert "another_function" in func_names

    def test_class_chunks_created(self, chunker, temp_python_file):
        """Test that class chunks are created."""
        chunks = chunker.chunk_file(temp_python_file)

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) == 1

        class_chunk = class_chunks[0]
        assert class_chunk.name == "TestClass"
        assert "Test class docstring" in class_chunk.content

    def test_method_chunks_created(self, chunker, temp_python_file):
        """Test that method chunks are created."""
        chunks = chunker.chunk_file(temp_python_file)

        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) == 1

        method_chunk = method_chunks[0]
        assert method_chunk.name == "method_one"
        assert method_chunk.metadata["class_name"] == "TestClass"

    def test_chunk_metadata(self, chunker, temp_python_file):
        """Test that chunk metadata is populated correctly."""
        chunks = chunker.chunk_file(temp_python_file)

        for chunk in chunks:
            assert chunk.file_path == str(temp_python_file)
            assert chunk.line_range[0] > 0
            assert chunk.line_range[1] >= chunk.line_range[0]
            assert chunk.metadata is not None

    def test_empty_file(self, chunker):
        """Test chunking an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) == 0
        finally:
            temp_path.unlink()

    def test_file_with_only_imports(self, chunker):
        """Test file with only imports."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import os\n")
            temp_path = Path(f.name)

        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) == 1
            assert chunks[0].chunk_type == "module"
        finally:
            temp_path.unlink()
