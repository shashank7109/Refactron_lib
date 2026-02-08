"""Tests for the RAG retriever module."""

import tempfile
from pathlib import Path

import pytest
from unittest.mock import Mock, MagicMock, patch

# Create a comprehensive mock for transformers
transformers_mock = MagicMock()
transformers_mock.__path__ = []

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
    from refactron.rag.retriever import ContextRetriever, RetrievedContext


class TestContextRetriever:
    """Test cases for ContextRetriever."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / ".rag").mkdir()
            yield workspace_path

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.retriever.chromadb")
    @patch("refactron.rag.retriever.Settings")
    @patch("refactron.rag.retriever.SentenceTransformer")
    def test_retriever_initialization(
        self, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test retriever initialization."""
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        retriever = ContextRetriever(temp_workspace)

        assert retriever.workspace_path == temp_workspace
        assert retriever.index_path == temp_workspace / ".rag"

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", False)
    def test_retriever_requires_chromadb(self, temp_workspace):
        """Test that retriever requires ChromaDB."""
        with pytest.raises(RuntimeError, match="ChromaDB is not available"):
            ContextRetriever(temp_workspace)

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.retriever.chromadb")
    @patch("refactron.rag.retriever.Settings")
    @patch("refactron.rag.retriever.SentenceTransformer")
    def test_retriever_missing_index(
        self, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test that missing index raises error."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        with pytest.raises(RuntimeError, match="Index not found"):
            ContextRetriever(temp_workspace)

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.retriever.chromadb")
    @patch("refactron.rag.retriever.Settings")
    @patch("refactron.rag.retriever.SentenceTransformer")
    def test_retrieve_similar(self, mock_transformer, mock_settings, mock_chroma, temp_workspace):
        """Test retrieving similar code chunks."""
        # Setup mock collection with results
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["def test(): pass"]],
            "metadatas": [
                [
                    {
                        "file_path": "/test.py",
                        "chunk_type": "function",
                        "name": "test",
                        "line_start": 1,
                        "line_end": 1,
                    }
                ]
            ],
            "distances": [[0.15]],
        }

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_model

        retriever = ContextRetriever(temp_workspace)
        results = retriever.retrieve_similar("test function", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedContext)
        assert results[0].name == "test"
        assert results[0].chunk_type == "function"

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.retriever.chromadb")
    @patch("refactron.rag.retriever.Settings")
    @patch("refactron.rag.retriever.SentenceTransformer")
    def test_retrieve_by_file(self, mock_transformer, mock_settings, mock_chroma, temp_workspace):
        """Test retrieving chunks by file path."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "documents": ["def test(): pass"],
            "metadatas": [
                {
                    "file_path": "/test.py",
                    "chunk_type": "function",
                    "name": "test",
                    "line_start": 1,
                    "line_end": 1,
                }
            ],
        }

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_transformer.return_value = mock_model

        retriever = ContextRetriever(temp_workspace)
        results = retriever.retrieve_by_file("/test.py")

        assert len(results) == 1
        assert results[0].file_path == "/test.py"

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.retriever.chromadb")
    @patch("refactron.rag.retriever.Settings")
    @patch("refactron.rag.retriever.SentenceTransformer")
    def test_retrieve_functions(self, mock_transformer, mock_settings, mock_chroma, temp_workspace):
        """Test retrieving only function chunks."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_model

        retriever = ContextRetriever(temp_workspace)
        retriever.retrieve_functions("test", top_k=5)

        # Verify that chunk_type filter was used
        call_args = mock_collection.query.call_args
        assert call_args.kwargs.get("where") == {"chunk_type": "function"}

    @patch("refactron.rag.retriever.CHROMA_AVAILABLE", True)
    @patch("refactron.rag.retriever.chromadb")
    @patch("refactron.rag.retriever.Settings")
    @patch("refactron.rag.retriever.SentenceTransformer")
    def test_retrieve_no_results(
        self, mock_transformer, mock_settings, mock_chroma, temp_workspace
    ):
        """Test retrieval with no results."""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_model = Mock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_model

        retriever = ContextRetriever(temp_workspace)
        results = retriever.retrieve_similar("nonexistent", top_k=5)

        assert len(results) == 0
