"""Context retrieval from the RAG index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    SentenceTransformer = None
    CHROMA_AVAILABLE = False


@dataclass
class RetrievedContext:
    """Represents a retrieved code context."""

    content: str
    file_path: str
    chunk_type: str
    name: str
    line_range: tuple
    distance: float  # Similarity distance
    metadata: dict


class ContextRetriever:
    """Retrieves relevant code context from the RAG index."""

    def __init__(
        self,
        workspace_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "code_chunks",
    ):
        """Initialize the context retriever.

        Args:
            workspace_path: Path to the workspace directory
            embedding_model: Name of the sentence-transformers model
            collection_name: Name of the ChromaDB collection
        """
        if not CHROMA_AVAILABLE:
            raise RuntimeError(
                "ChromaDB is not available. Install with: pip install chromadb sentence-transformers"
            )

        self.workspace_path = Path(workspace_path)
        self.index_path = self.workspace_path / ".rag"

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.index_path / "chroma"), settings=Settings(anonymized_telemetry=False)
        )

        # Get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            raise RuntimeError(
                f"Index not found at {self.index_path}. Run 'refactron rag index' first."
            )

    def retrieve_similar(
        self, query: str, top_k: int = 5, chunk_type: Optional[str] = None
    ) -> List[RetrievedContext]:
        """Retrieve similar code chunks.

        Args:
            query: The search query
            top_k: Number of results to return
            chunk_type: Optional filter by chunk type (function/class/module)

        Returns:
            List of retrieved contexts sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]

        # Build query filters
        where_filter = {}
        if chunk_type:
            where_filter["chunk_type"] = chunk_type

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None,
        )

        # Parse results
        contexts = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0.0

                contexts.append(
                    RetrievedContext(
                        content=doc,
                        file_path=metadata.get("file_path", ""),
                        chunk_type=metadata.get("chunk_type", ""),
                        name=metadata.get("name", ""),
                        line_range=(metadata.get("line_start", 0), metadata.get("line_end", 0)),
                        distance=distance,
                        metadata=metadata,
                    )
                )

        return contexts

    def retrieve_by_file(self, file_path: str) -> List[RetrievedContext]:
        """Retrieve all chunks from a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of all chunks from the file
        """
        # Query by file path metadata
        results = self.collection.get(where={"file_path": file_path})

        # Parse results
        contexts = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i]

                contexts.append(
                    RetrievedContext(
                        content=doc,
                        file_path=metadata.get("file_path", ""),
                        chunk_type=metadata.get("chunk_type", ""),
                        name=metadata.get("name", ""),
                        line_range=(metadata.get("line_start", 0), metadata.get("line_end", 0)),
                        distance=0.0,  # Not applicable for exact match
                        metadata=metadata,
                    )
                )

        return contexts

    def retrieve_functions(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        """Retrieve similar functions.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of similar function chunks
        """
        return self.retrieve_similar(query, top_k=top_k, chunk_type="function")

    def retrieve_classes(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        """Retrieve similar classes.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of similar class chunks
        """
        return self.retrieve_similar(query, top_k=top_k, chunk_type="class")
