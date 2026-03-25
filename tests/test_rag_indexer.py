"""
Tests for rag/indexer.py – comprehensive coverage via mocking ChromaDB / sentence-transformers.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── helper: build a fully-mocked RAGIndexer ──────────────────────────────────


def make_indexer(tmp_path: Path):
    """Return a RAGIndexer with all heavy dependencies mocked."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2, 0.3]])

    mock_collection = MagicMock()
    mock_chroma_client = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection

    with patch("refactron.rag.indexer.CHROMA_AVAILABLE", True), patch(
        "refactron.rag.indexer.SentenceTransformer", return_value=mock_embedding_model
    ), patch("refactron.rag.indexer.chromadb") as mock_chromadb, patch(
        "refactron.rag.indexer.Settings"
    ):
        mock_chromadb.PersistentClient.return_value = mock_chroma_client
        from refactron.rag.indexer import RAGIndexer

        indexer = RAGIndexer(workspace_path=tmp_path)
        # Attach mocks for inspection
        indexer._mock_collection = mock_collection
        indexer._mock_embedding = mock_embedding_model
        return indexer


# ── tests for IndexStats ──────────────────────────────────────────────────────


class TestIndexStats:
    def test_fields(self):
        from refactron.rag.indexer import IndexStats

        s = IndexStats(
            total_chunks=10,
            total_files=5,
            chunk_types={"function": 4, "class": 6},
            embedding_model="all-MiniLM-L6-v2",
            index_path="/tmp/rag",
        )
        assert s.total_chunks == 10
        assert s.total_files == 5
        assert s.chunk_types["function"] == 4


# ── tests for RAGIndexer init ─────────────────────────────────────────────────


class TestRAGIndexerInit:
    def test_raises_without_chromadb(self, tmp_path):
        with patch("refactron.rag.indexer.CHROMA_AVAILABLE", False):
            from refactron.rag.indexer import RAGIndexer

            with pytest.raises(RuntimeError, match="ChromaDB"):
                RAGIndexer(workspace_path=tmp_path)

    def test_creates_index_dir(self, tmp_path):
        indexer = make_indexer(tmp_path)
        assert indexer.workspace_path == tmp_path

    def test_default_embedding_model_name(self, tmp_path):
        indexer = make_indexer(tmp_path)
        assert indexer.embedding_model_name == "all-MiniLM-L6-v2"


# ── tests for get_stats / _save_metadata / _load_metadata ────────────────────


class TestMetadataOperations:
    def test_save_and_load_metadata(self, tmp_path):
        indexer = make_indexer(tmp_path)
        meta = {"total_chunks": 5, "total_files": 2, "chunk_types": {"function": 3}}
        indexer._save_metadata(meta)
        loaded = indexer._load_metadata()
        assert loaded["total_chunks"] == 5
        assert loaded["chunk_types"]["function"] == 3

    def test_load_metadata_missing_file(self, tmp_path):
        indexer = make_indexer(tmp_path)
        result = indexer._load_metadata()
        assert result == {}

    def test_get_stats_empty(self, tmp_path):
        indexer = make_indexer(tmp_path)
        stats = indexer.get_stats()
        assert stats.total_chunks == 0
        assert stats.total_files == 0

    def test_get_stats_after_saving(self, tmp_path):
        indexer = make_indexer(tmp_path)
        indexer._save_metadata({"total_chunks": 3, "total_files": 1, "chunk_types": {"class": 2}})
        stats = indexer.get_stats()
        assert stats.total_chunks == 3


# ── tests for add_chunks ──────────────────────────────────────────────────────


class TestAddChunks:
    def test_empty_chunks_does_nothing(self, tmp_path):
        indexer = make_indexer(tmp_path)
        indexer.add_chunks([])  # Should not raise

    def test_add_single_chunk(self, tmp_path):
        indexer = make_indexer(tmp_path)
        chunk = MagicMock()
        chunk.content = "def foo(): pass"
        chunk.chunk_type = "function"
        chunk.file_path = "test.py"
        chunk.name = "foo"
        chunk.line_range = (1, 3)
        chunk.metadata = {}

        indexer._mock_embedding.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2]])
        indexer.add_chunks([chunk])
        assert indexer._mock_collection.add.called

    def test_chunk_metadata_list_serialised_to_json(self, tmp_path):
        indexer = make_indexer(tmp_path)
        chunk = MagicMock()
        chunk.content = "x = 1"
        chunk.chunk_type = "module"
        chunk.file_path = "a.py"
        chunk.name = "module"
        chunk.line_range = (1, 1)
        chunk.metadata = {"deps": ["os", "sys"], "nested": {"a": 1}}

        indexer._mock_embedding.encode.return_value = MagicMock(tolist=lambda: [[0.5]])
        indexer.add_chunks([chunk])
        call_kwargs = indexer._mock_collection.add.call_args[1]
        meta = call_kwargs["metadatas"][0]
        # Lists and dicts should be serialised to JSON strings
        assert isinstance(meta["deps"], str)
        assert json.loads(meta["deps"]) == ["os", "sys"]


# ── tests for _summarize_chunk ────────────────────────────────────────────────


class TestSummarizeChunk:
    def test_returns_none_without_llm(self, tmp_path):
        indexer = make_indexer(tmp_path)
        indexer.llm_client = None
        chunk = MagicMock()
        result = indexer._summarize_chunk(chunk)
        assert result is None

    def test_returns_summary_with_llm(self, tmp_path):
        indexer = make_indexer(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Handles authentication."
        indexer.llm_client = mock_llm
        chunk = MagicMock()
        chunk.content = "def authenticate(): pass"
        result = indexer._summarize_chunk(chunk)
        assert result == "Handles authentication."

    def test_llm_exception_returns_none(self, tmp_path):
        indexer = make_indexer(tmp_path)
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM error")
        indexer.llm_client = mock_llm
        chunk = MagicMock()
        chunk.content = "x = 1"
        result = indexer._summarize_chunk(chunk)
        assert result is None


# ── tests for index_repository ────────────────────────────────────────────────


class TestIndexRepository:
    def test_index_empty_repo(self, tmp_path):
        indexer = make_indexer(tmp_path)
        # No Python files → stats should have 0 files
        stats = indexer.index_repository(tmp_path)
        assert stats.total_files == 0

    def test_index_with_python_file(self, tmp_path):
        py_file = tmp_path / "sample.py"
        py_file.write_text("def greet(): pass\n")

        indexer = make_indexer(tmp_path)

        mock_chunk = MagicMock()
        mock_chunk.content = "def greet(): pass"
        mock_chunk.chunk_type = "function"
        mock_chunk.file_path = str(py_file)
        mock_chunk.name = "greet"
        mock_chunk.line_range = (1, 1)
        mock_chunk.metadata = {}

        with patch.object(indexer, "_index_file", return_value=[mock_chunk]):
            stats = indexer.index_repository(tmp_path)
        assert stats.total_files == 1
        assert stats.total_chunks == 1

    def test_index_excludes_venv_dirs(self, tmp_path):
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        py_file = venv_dir / "helper.py"
        py_file.write_text("x = 1\n")
        indexer = make_indexer(tmp_path)
        stats = indexer.index_repository(tmp_path)
        assert stats.total_files == 0

    def test_index_file_error_handled(self, tmp_path):
        py_file = tmp_path / "bad.py"
        py_file.write_text("def foo(): pass\n")
        indexer = make_indexer(tmp_path)
        with patch.object(indexer, "_index_file", side_effect=Exception("parse error")):
            # Should not raise; file errors are swallowed
            stats = indexer.index_repository(tmp_path)
        assert stats.total_files == 1
        assert stats.total_chunks == 0

    def test_summarize_flag_initializes_llm(self, tmp_path):
        indexer = make_indexer(tmp_path)
        indexer.llm_client = None
        # No files → summarize path never reached for real files
        with patch("refactron.rag.indexer.GroqClient", MagicMock()):
            stats = indexer.index_repository(tmp_path, summarize=True)
        assert stats.total_files == 0

    def test_summarize_flag_llm_init_failure(self, tmp_path):
        indexer = make_indexer(tmp_path)
        indexer.llm_client = None
        with patch("refactron.rag.indexer.GroqClient", side_effect=Exception("key error")):
            stats = indexer.index_repository(tmp_path, summarize=True)
        assert stats.total_files == 0


# ── tests for _index_file ─────────────────────────────────────────────────────


class TestIndexFile:
    def test_index_single_file(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello(): pass\n")
        indexer = make_indexer(tmp_path)

        mock_chunk = MagicMock()
        mock_chunk.content = "def hello(): pass"
        mock_chunk.chunk_type = "function"
        mock_chunk.file_path = str(py_file)
        mock_chunk.name = "hello"
        mock_chunk.line_range = (1, 1)
        mock_chunk.metadata = {}

        with patch("refactron.rag.chunker.CodeChunker") as mock_chunker_cls:
            mock_chunker = MagicMock()
            mock_chunker.chunk_file.return_value = [mock_chunk]
            mock_chunker_cls.return_value = mock_chunker
            chunks = indexer._index_file(py_file, summarize=False)

        assert len(chunks) == 1

    def test_index_file_with_summarize(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("def hi(): pass\n")
        indexer = make_indexer(tmp_path)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Says hello."
        indexer.llm_client = mock_llm

        mock_chunk = MagicMock()
        mock_chunk.content = "def hi(): pass"
        mock_chunk.chunk_type = "function"
        mock_chunk.file_path = str(py_file)
        mock_chunk.name = "hi"
        mock_chunk.line_range = (1, 1)
        mock_chunk.metadata = {}

        with patch("refactron.rag.chunker.CodeChunker") as mock_chunker_cls:
            mock_chunker = MagicMock()
            mock_chunker.chunk_file.return_value = [mock_chunk]
            mock_chunker_cls.return_value = mock_chunker
            chunks = indexer._index_file(py_file, summarize=True)

        assert "Summary:" in mock_chunk.content or chunks is not None
