# refactron.rag

RAG (Retrieval-Augmented Generation) infrastructure for code indexing and retrieval.

## Classes

## Functions


---

# refactron.rag.chunker

Code chunking strategies for RAG indexing.

## Classes

### CodeChunk

```python
CodeChunk(content: 'str', chunk_type: 'str', file_path: 'str', line_range: 'Tuple[int, int]', name: 'str', dependencies: 'List[str]', metadata: 'Dict[str, Any]') -> None
```

Represents a semantic chunk of code.

#### CodeChunk.__init__

```python
CodeChunk.__init__(self, content: 'str', chunk_type: 'str', file_path: 'str', line_range: 'Tuple[int, int]', name: 'str', dependencies: 'List[str]', metadata: 'Dict[str, Any]') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### CodeChunker

```python
CodeChunker(parser: 'CodeParser')
```

Chunks parsed code into semantic units for embedding.

#### CodeChunker.__init__

```python
CodeChunker.__init__(self, parser: 'CodeParser')
```

Initialize the chunker.

Args:
    parser: CodeParser instance for parsing files

#### CodeChunker.chunk_file

```python
CodeChunker.chunk_file(self, file_path: 'Path') -> 'List[CodeChunk]'
```

Chunk a file into semantic units.

Args:
    file_path: Path to the Python file

Returns:
    List of code chunks

## Functions


---

# refactron.rag.indexer

Vector index management using ChromaDB.

## Classes

### IndexStats

```python
IndexStats(total_chunks: 'int', total_files: 'int', chunk_types: 'dict', embedding_model: 'str', index_path: 'str') -> None
```

Statistics about the RAG index.

#### IndexStats.__init__

```python
IndexStats.__init__(self, total_chunks: 'int', total_files: 'int', chunk_types: 'dict', embedding_model: 'str', index_path: 'str') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### RAGIndexer

```python
RAGIndexer(workspace_path: 'Path', embedding_model: 'str' = 'all-MiniLM-L6-v2', collection_name: 'str' = 'code_chunks', llm_client: 'Optional[GroqClient]' = None)
```

Manages code indexing for RAG retrieval.

#### RAGIndexer.__init__

```python
RAGIndexer.__init__(self, workspace_path: 'Path', embedding_model: 'str' = 'all-MiniLM-L6-v2', collection_name: 'str' = 'code_chunks', llm_client: 'Optional[GroqClient]' = None)
```

Initialize the RAG indexer.

Args:
    workspace_path: Path to the workspace directory
    embedding_model: Name of the sentence-transformers model
    collection_name: Name of the ChromaDB collection
    llm_client: Optional LLM client for code summarization

#### RAGIndexer.add_chunks

```python
RAGIndexer.add_chunks(self, chunks: 'List[CodeChunk]') -> 'None'
```

Add code chunks to the vector index.

Args:
    chunks: List of code chunks to add

#### RAGIndexer.get_stats

```python
RAGIndexer.get_stats(self) -> 'IndexStats'
```

Get statistics about the current index.

Returns:
    Index statistics

#### RAGIndexer.index_repository

```python
RAGIndexer.index_repository(self, repo_path: 'Optional[Path]' = None, summarize: 'bool' = False) -> 'IndexStats'
```

Index an entire repository.

Args:
    repo_path: Path to repository (defaults to workspace_path)
    summarize: Whether to use AI to summarize code for better retrieval

Returns:
    Statistics about the indexed content

## Functions


---

# refactron.rag.parser

Code parser using tree-sitter for AST-aware code analysis.

## Classes

### CodeParser

```python
CodeParser()
```

AST-aware code parser using tree-sitter.

#### CodeParser.__init__

```python
CodeParser.__init__(self)
```

Initialize the parser.

#### CodeParser.parse_file

```python
CodeParser.parse_file(self, file_path: 'Path') -> 'ParsedFile'
```

Parse a Python file.

Args:
    file_path: Path to the Python file

Returns:
    ParsedFile object containing all parsed elements

### ParsedClass

```python
ParsedClass(name: 'str', body: 'str', docstring: 'Optional[str]', line_range: 'Tuple[int, int]', methods: 'List[ParsedFunction]') -> None
```

Represents a parsed class.

#### ParsedClass.__init__

```python
ParsedClass.__init__(self, name: 'str', body: 'str', docstring: 'Optional[str]', line_range: 'Tuple[int, int]', methods: 'List[ParsedFunction]') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### ParsedFile

```python
ParsedFile(file_path: 'str', imports: 'List[str]', functions: 'List[ParsedFunction]', classes: 'List[ParsedClass]', module_docstring: 'Optional[str]') -> None
```

Represents a parsed Python file.

#### ParsedFile.__init__

```python
ParsedFile.__init__(self, file_path: 'str', imports: 'List[str]', functions: 'List[ParsedFunction]', classes: 'List[ParsedClass]', module_docstring: 'Optional[str]') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### ParsedFunction

```python
ParsedFunction(name: 'str', body: 'str', docstring: 'Optional[str]', line_range: 'Tuple[int, int]', params: 'List[str]') -> None
```

Represents a parsed function.

#### ParsedFunction.__init__

```python
ParsedFunction.__init__(self, name: 'str', body: 'str', docstring: 'Optional[str]', line_range: 'Tuple[int, int]', params: 'List[str]') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions


---

# refactron.rag.retriever

Context retrieval from the RAG index.

## Classes

### ContextRetriever

```python
ContextRetriever(workspace_path: 'Path', embedding_model: 'str' = 'all-MiniLM-L6-v2', collection_name: 'str' = 'code_chunks')
```

Retrieves relevant code context from the RAG index.

#### ContextRetriever.__init__

```python
ContextRetriever.__init__(self, workspace_path: 'Path', embedding_model: 'str' = 'all-MiniLM-L6-v2', collection_name: 'str' = 'code_chunks')
```

Initialize the context retriever.

Args:
    workspace_path: Path to the workspace directory
    embedding_model: Name of the sentence-transformers model
    collection_name: Name of the ChromaDB collection

#### ContextRetriever.retrieve_by_file

```python
ContextRetriever.retrieve_by_file(self, file_path: 'str') -> 'List[RetrievedContext]'
```

Retrieve all chunks from a specific file.

Args:
    file_path: Path to the file

Returns:
    List of all chunks from the file

#### ContextRetriever.retrieve_classes

```python
ContextRetriever.retrieve_classes(self, query: 'str', top_k: 'int' = 5) -> 'List[RetrievedContext]'
```

Retrieve similar classes.

Args:
    query: The search query
    top_k: Number of results to return

Returns:
    List of similar class chunks

#### ContextRetriever.retrieve_functions

```python
ContextRetriever.retrieve_functions(self, query: 'str', top_k: 'int' = 5) -> 'List[RetrievedContext]'
```

Retrieve similar functions.

Args:
    query: The search query
    top_k: Number of results to return

Returns:
    List of similar function chunks

#### ContextRetriever.retrieve_similar

```python
ContextRetriever.retrieve_similar(self, query: 'str', top_k: 'int' = 5, chunk_type: 'Optional[str]' = None) -> 'List[RetrievedContext]'
```

Retrieve similar code chunks.

Args:
    query: The search query
    top_k: Number of results to return
    chunk_type: Optional filter by chunk type (function/class/module)

Returns:
    List of retrieved contexts sorted by relevance

### RetrievedContext

```python
RetrievedContext(content: 'str', file_path: 'str', chunk_type: 'str', name: 'str', line_range: 'tuple', distance: 'float', metadata: 'dict') -> None
```

Represents a retrieved code context.

#### RetrievedContext.__init__

```python
RetrievedContext.__init__(self, content: 'str', file_path: 'str', chunk_type: 'str', name: 'str', line_range: 'tuple', distance: 'float', metadata: 'dict') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions
