"""
Refactron CLI - RAG Module.
Commands for RAG (Retrieval-Augmented Generation) management.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.panel import Panel

from refactron.cli.ui import _auth_banner, console
from refactron.cli.utils import _setup_logging
from refactron.core.workspace import WorkspaceManager
from refactron.rag.indexer import RAGIndexer
from refactron.rag.retriever import ContextRetriever


@click.group()
def rag() -> None:
    """RAG (Retrieval-Augmented Generation) management commands."""
    pass


@rag.command("index")
@click.option(
    "--background",
    is_flag=True,
    help="Run in background mode (suppress output)",
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Use AI to summarize code for better retrieval",
)
def rag_index(background: bool, summarize: bool) -> None:
    """Index the current workspace for RAG retrieval."""
    if background:
        # Suppress all logging and output in background mode
        logging.getLogger().setLevel(logging.CRITICAL)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    else:
        _setup_logging()
        console.print()
        _auth_banner("Index Repository")
        console.print()

    workspace_mgr = WorkspaceManager()

    # Get current workspace
    current_workspace = workspace_mgr.get_workspace_by_path(str(Path.cwd()))
    if not current_workspace:
        if not background:
            console.print(
                "[red]Error: Not in a connected workspace.[/red]\n\n"
                "[dim]Run 'refactron repo connect <repo-name>' first.[/dim]"
            )
        raise SystemExit(1)

    local_path = Path(current_workspace.local_path)

    if not background:
        console.print(f"[primary]Indexing:[/primary] {current_workspace.repo_full_name}\n")

    try:
        if background:
            # Run without visual feedback
            indexer = RAGIndexer(local_path)
            indexer.index_repository(local_path, summarize=summarize)
        else:
            with console.status("[primary]Parsing and indexing code...[/primary]"):
                indexer = RAGIndexer(local_path)
                stats = indexer.index_repository(local_path, summarize=summarize)

            console.print(
                Panel(
                    f"[success]Indexing complete![/success]\n\n"
                    f"Files indexed: [bold]{stats.total_files}[/bold]\n"
                    f"Code chunks: [bold]{stats.total_chunks}[/bold]\n"
                    f"Index location: [dim]{stats.index_path}[/dim]\n\n"
                    f"[dim]Chunk breakdown:[/dim]\n"
                    f"  • Functions: {stats.chunk_types.get('function', 0)}\n"
                    f"  • Classes: {stats.chunk_types.get('class', 0)}\n"
                    f"  • Methods: {stats.chunk_types.get('method', 0)}\n"
                    f"  • Modules: {stats.chunk_types.get('module', 0)}",
                    title="✓ Indexed",
                    border_style="success",
                    box=box.ROUNDED,
                )
            )
    except Exception as e:
        console.print(f"[red]Error indexing repository: {e}[/red]")
        raise SystemExit(1)


@rag.command("search")
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results to return")
@click.option("--type", "chunk_type", help="Filter by chunk type (function/class/module)")
@click.option(
    "--rerank",
    is_flag=True,
    help="Use AI to rerank results for better accuracy",
)
def rag_search(query: str, top_k: int, chunk_type: Optional[str], rerank: bool) -> None:
    """Search the RAG index for similar code."""
    _setup_logging()
    console.print()

    workspace_mgr = WorkspaceManager()

    # Get current workspace
    current_workspace = workspace_mgr.get_workspace_by_path(str(Path.cwd()))
    if not current_workspace:
        console.print(
            "[red]Error: Not in a connected workspace.[/red]\n\n"
            "[dim]Run 'refactron repo connect <repo-name>' first.[/dim]"
        )
        raise SystemExit(1)

    local_path = Path(current_workspace.local_path)

    try:
        retriever = ContextRetriever(local_path)
        results = retriever.retrieve_similar(query, top_k=top_k, chunk_type=chunk_type)

        if not results:
            console.print(f"[yellow]No results found for: {query}[/yellow]")
            return

        console.print(f"\n[primary]Found {len(results)} results for:[/primary] {query}\n")

        for i, result in enumerate(results, 1):
            relevance_score = max(0, 1 - result.distance) * 100

            # AI Reranking if enabled
            if rerank:
                try:
                    from refactron.llm.client import GroqClient

                    client = GroqClient()
                    prompt = (  # noqa: E501
                        f"Rate the relevance of the following code snippet to the user query: '{query}'\n\n"  # noqa: E501
                        f"Code:\n{result.content[:500]}\n\n"
                        "Provide only a percentage number (e.g. 85%) representing how well this code matches "  # noqa: E501
                        "the semantic intent of the query."
                    )
                    ai_response = client.generate(
                        prompt=prompt,
                        system="You are a code relevance evaluator. Output only the percentage.",
                        max_tokens=10,
                    )
                    # Extract number from response (e.g. "85%" or "85")
                    import re

                    match = re.search(r"(\d+)%", ai_response) or re.search(r"(\d+)", ai_response)
                    if match:
                        relevance_score = float(match.group(1))
                except Exception:
                    pass  # Fallback to distance-based score

            console.print(
                Panel(
                    f"[bold]{result.name}[/bold] ({result.chunk_type})\n"
                    f"[dim]{result.file_path}:{result.line_range[0]}-{result.line_range[1]}[/dim]\n\n"  # noqa: E501
                    f"```python\n{result.content[:200]}{'...' if len(result.content) > 200 else ''}\n```\n\n"  # noqa: E501
                    f"[dim]Similarity: {relevance_score / 100.0:.2%}[/dim]",
                    title=f"Result {i}/{len(results)}",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]\n")
        console.print("[dim]Run 'refactron rag index' to create an index first.[/dim]")
        raise SystemExit(1)


@rag.command("status")
def rag_status() -> None:
    """Show RAG index statistics."""
    _setup_logging()
    console.print()

    workspace_mgr = WorkspaceManager()

    # Get current workspace
    current_workspace = workspace_mgr.get_workspace_by_path(str(Path.cwd()))
    if not current_workspace:
        console.print(
            "[red]Error: Not in a connected workspace.[/red]\n\n"
            "[dim]Run 'refactron repo connect <repo-name>' first.[/dim]"
        )
        raise SystemExit(1)

    local_path = Path(current_workspace.local_path)

    try:
        indexer = RAGIndexer(local_path)
        stats = indexer.get_stats()

        if stats.total_chunks == 0:
            console.print(
                "[yellow]No index found.[/yellow]\n\n"
                "[dim]Run 'refactron rag index' to create one.[/dim]"
            )
            return

        console.print(
            Panel(
                f"[primary]RAG Index Status[/primary]\n\n"
                f"Files indexed: [bold]{stats.total_files}[/bold]\n"
                f"Total chunks: [bold]{stats.total_chunks}[/bold]\n"
                f"Embedding model: [dim]{stats.embedding_model}[/dim]\n"
                f"Index location: [dim]{stats.index_path}[/dim]\n\n"
                f"[dim]Chunk breakdown:[/dim]\n"
                f"  • Functions: {stats.chunk_types.get('function', 0)}\n"
                f"  • Classes: {stats.chunk_types.get('class', 0)}\n"
                f"  • Methods: {stats.chunk_types.get('method', 0)}\n"
                f"  • Modules: {stats.chunk_types.get('module', 0)}",
                title="RAG Status",
                border_style="primary",
                box=box.ROUNDED,
            )
        )
    except Exception as e:
        console.print(f"[yellow]No index found: {e}[/yellow]\n")
        console.print("[dim]Run 'refactron rag index' to create one.[/dim]")
