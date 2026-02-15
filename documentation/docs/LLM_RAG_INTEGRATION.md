# LLM & RAG Integration Guide

**Harness the power of Large Language Models for intelligent code refactoring and documentation.**

---

## Overview

In version v1.0.15, Refactron introduces a powerful AI-driven subsystem that combines LLM reasoning with RAG (Retrieval-Augmented Generation). This allows Refactron to understand your entire project context when suggesting refactorings or generating documentation.

### Core Components

1.  **LLM Orchestrator**: Coordinates between the code analyzer, retriever, and LLM backends.
2.  **RAG System**: Uses a vector database (ChromaDB) to index and retrieve relevant code chunks.
3.  **LLM Backends**: Support for high-performance providers like Groq, as well as local or custom backends.
4.  **Safety Gate**: Ensures that LLM-generated code adheres to safety standards and doesn't introduce syntax errors.

---

## Getting Started

### 1. Prerequisites

- Python 3.8+
- ChromaDB (`pip install chromadb`)
- Sentence Transformers (`pip install sentence-transformers`)
- LLM API Key (e.g., Groq API Key)

### 2. Configuration

Set your LLM API key as an environment variable:

```bash
export GROQ_API_KEY='your-api-key-here'
```

Alternatively, configure it in your `.refactron.yaml`:

```yaml
llm:
  provider: groq
  model: llama3-70b-8192
  temperature: 0.1
  max_tokens: 4096

rag:
  enabled: true
  storage_dir: .refactron/rag_index
  embedding_model: all-MiniLM-L6-v2
```

---

## Usage

### Indexing your Project

Before using RAG-powered features, you need to index your repository:

```bash
refactron rag index
```

This will parse your Python files and store embeddings in the local vector database.

### AI-Powered Refactoring

When you run the `refactor` command, Refactron can now use the LLM to generate more sophisticated suggestions:

```bash
refactron refactor myfile.py --ai --preview
```

The `--ai` flag enables LLM-based suggestion generation, which uses retrieved context from your project to provide more accurate fixes.

### Generating Documentation

Generate comprehensive docstrings and technical documentation using the LLM:

```bash
refactron docs generate myfile.py
```

Refactron will analyze the code structure and use the LLM to write high-quality documentation that follows PEP 257 or your configured style.

---

## Technical Details

### RAG Workflow

1.  **Parsing**: Code files are parsed into semantic chunks (classes, methods, functions).
2.  **Embedding**: Chunks are converted into vector representations using `sentence-transformers`.
3.  **Indexing**: Vectors and metadata are stored in `ChromaDB`.
4.  **Retrieval**: When an issue is analyzed, the system retrieves the most relevant code chunks as context for the LLM.

### LLM Orchestration

The `LLMOrchestrator` handles the prompt engineering, ensuring that the LLM receives the right balance of task instructions and code context. It also includes a JSON cleaning layer to reliably parse LLM outputs.

---

## Best Practices

-   **Keep the Index Updated**: Re-run `refactron rag index` after significant code changes.
-   **Model Selection**: Higher-parameter models (like Llama 3 70B) generally provide better refactoring logic but may be slower.
-   **Review AI Suggestions**: Always use `--preview` to review AI-generated code before applying it.

---

**Refactron AI** - Bringing semantic understanding to code refactoring! 🚀🤖
