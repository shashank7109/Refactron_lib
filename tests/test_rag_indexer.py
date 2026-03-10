"""Tests for the RAG indexer module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, create_autospec, patch

import pytest

# Create a comprehensive mock for transformers that handles all submodule access
transformers_mock = MagicMock()
transformers_mock.__path__ = []  # Make it look like a package

# Patch sys.modules before importing anything that depends on sentence-transformers
with patch.dict(
    "sys.modules",
    {
        "transformers": transformers_mock,
        "transformers.configuration_utils": MagicMock(),
        "transformers.utils": MagicMock(),
        "transformers.models": MagicMock(),
        "transformers.file_utils": MagicMock(),
        "transformers.tokenization_utils_base": MagicMock(),
    },
):
    from refactron.rag.chunker import CodeChunk


class TestRAGIndexer:
    """Test cases for RAGIndexer."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            # Create sample Python files
            (workspace_path / "simple.py").write_text(
                '''
"""Simple module."""

def hello():
    """Say hello."""
    return "Hello"
'''
            )

            (workspace_path / "utils.py").write_text(
                '''
"""Utility functions."""

class Calculator:
    """A simple calculator."""

    def add(self, x, y):
        """Add two numbers."""
        return x + y
'''
            )

            yield workspace_path

    @patch("refactron.rag.indexer.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.indexer.chromadb")
    @patch("refactron.rag.indexer.Settings")
    @patch("refactron.rag.indexer.SentenceTransformer")
    @patch("refactron.rag.indexer.CodeParser")
    def test_indexer_initialization(
        self, mock_parser, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test indexer initialization."""
        from refactron.rag.indexer import RAGIndexer

        indexer = RAGIndexer(temp_workspace)

        assert indexer.workspace_path == temp_workspace
        assert indexer.index_path == temp_workspace / ".rag"
        assert indexer.embedding_model_name == "all-MiniLM-L6-v2"

    @patch("refactron.rag.indexer.CHROMA_AVAILABLE", False)
    def test_indexer_requires_chromadb(self, temp_workspace):
        """Test that indexer requires ChromaDB."""
        from refactron.rag.indexer import RAGIndexer

        with pytest.raises(RuntimeError, match="ChromaDB is not available"):
            RAGIndexer(temp_workspace)

    @patch("refactron.rag.indexer.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.indexer.chromadb")
    @patch("refactron.rag.indexer.Settings")
    @patch("refactron.rag.indexer.SentenceTransformer")
    @patch("refactron.rag.indexer.CodeParser")
    def test_add_chunks(
        self, mock_parser, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test adding chunks to the index."""
        from refactron.rag.indexer import RAGIndexer

        # Setup mocks
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_model.encode.return_value = Mock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        indexer = RAGIndexer(temp_workspace)

        # Create test chunk
        chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            file_path="/test/file.py",
            line_range=(1, 1),
            name="test",
            dependencies=[],
            metadata={},
        )

        indexer.add_chunks([chunk])

        # Verify chunk was added
        assert mock_collection.add.called

    @patch("refactron.rag.indexer.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.indexer.chromadb")
    @patch("refactron.rag.indexer.Settings")
    @patch("refactron.rag.indexer.SentenceTransformer")
    @patch("refactron.rag.indexer.CodeParser")
    def test_add_empty_chunks(
        self, mock_parser, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test that adding empty chunks list does nothing."""
        from refactron.rag.indexer import RAGIndexer

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        indexer = RAGIndexer(temp_workspace)
        indexer.add_chunks([])

        # Verify nothing was added
        assert not mock_collection.add.called

    @patch("refactron.rag.indexer.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.indexer.chromadb")
    @patch("refactron.rag.indexer.Settings")
    @patch("refactron.rag.indexer.SentenceTransformer")
    @patch("refactron.rag.indexer.CodeParser")
    def test_get_stats(
        self, mock_parser, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test getting index statistics."""
        from refactron.rag.indexer import RAGIndexer

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        indexer = RAGIndexer(temp_workspace)

        # Save some metadata
        indexer._save_metadata(
            {"total_chunks": 10, "total_files": 2, "chunk_types": {"function": 8, "class": 2}}
        )

        stats = indexer.get_stats()

        assert stats.total_chunks == 10
        assert stats.total_files == 2
        assert stats.chunk_types["function"] == 8
