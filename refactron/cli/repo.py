"""
Refactron CLI - Repository Module.
Commands for managing GitHub repository connections.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.panel import Panel
from rich.table import Table

from refactron.cli.ui import _auth_banner, console
from refactron.cli.utils import _setup_logging
from refactron.core.device_auth import DEFAULT_API_BASE_URL
from refactron.core.repositories import Repository, list_repositories
from refactron.core.workspace import WorkspaceManager, WorkspaceMapping


@click.group()
def repo() -> None:
    """Manage GitHub repository connections."""
    pass


@repo.command("list")
@click.option(
    "--api-base-url",
    default=DEFAULT_API_BASE_URL,
    show_default=True,
    help="Refactron API base URL",
)
def repo_list(api_base_url: str) -> None:
    """
    List all GitHub repositories connected to your account.

    Shows repositories that have been connected via the Refactron WebApp.
    """
    _setup_logging()
    console.print()
    _auth_banner("Repository List")
    console.print()

    try:
        with console.status("[primary]Fetching repositories...[/primary]"):
            repositories = list_repositories(api_base_url)

        if not repositories:
            console.print(
                Panel(
                    "[yellow]No repositories found.\n\n"
                    "Please connect your GitHub account on the Refactron website:[/yellow]\n"
                    f"[link]{api_base_url.replace('/api', '')}[/link]",
                    title="No Repositories",
                    border_style="warning",
                    box=box.ROUNDED,
                )
            )
            return

        # Create table
        table = Table(
            title=f"Connected Repositories ({len(repositories)})",
            show_header=True,
            header_style="primary",
            box=box.ROUNDED,
            border_style="panel.border",
        )
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="dim")
        table.add_column("Language", justify="center", style="green")
        table.add_column("Private", justify="center")
        table.add_column("Updated", style="dim", no_wrap=True)

        # Check which repos are already connected locally
        workspace_mgr = WorkspaceManager()

        for repository in repositories:
            workspace = workspace_mgr.get_workspace(repository.full_name)
            name_display = repository.name
            if workspace:
                name_display = f"✓ {repository.name}"

            desc = repository.description or "[dim]No description[/dim]"
            if len(desc) > 60:
                desc = desc[:57] + "..."

            lang = repository.language or "—"
            private = "Yes" if repository.private else "No"
            updated = repository.updated_at.split("T")[0]  # Just the date

            table.add_row(name_display, desc, lang, private, updated)

        console.print(table)
        console.print("\n[dim]✓ = Already connected locally[/dim]")
        console.print(
            "[dim]Tip: Use 'refactron repo connect' to link a repository to a local directory[/dim]"
        )

    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


@repo.command("connect")
@click.argument("repo_name", required=False)
@click.option(
    "--path",
    "-p",
    type=click.Path(file_okay=False),
    default=None,
    help="Local path to connect (default: auto-clone to managed workspace)",
)
@click.option(
    "--api-base-url",
    default=DEFAULT_API_BASE_URL,
    show_default=True,
    help="Refactron API base URL",
)
def repo_connect(repo_name: Optional[str], path: Optional[str], api_base_url: str) -> None:
    """
    Connect to a GitHub repository.

    REPO_NAME: Name of the repository (e.g., 'my-project' or 'user/my-project')

    If the repository doesn't exist locally, it will be cloned automatically
    to ~/.refactron/workspaces/<repo-name>/
    """
    _setup_logging()
    console.print()
    _auth_banner("Connect Repository")
    console.print()

    workspace_mgr = WorkspaceManager()

    # If path is provided, use existing behavior (map existing local directory)
    if path:
        local_path = Path(path).resolve()

        # Auto-detect repository if not provided
        if not repo_name:
            console.print("[dim]No repository specified, attempting auto-detection...[/dim]\n")
            detected = workspace_mgr.detect_repository(local_path)
            if detected:
                console.print(f"[success]Detected repository: {detected}[/success]\n")
                repo_name = detected
            else:
                console.print(
                    "[red]Could not auto-detect repository from .git config.[/red]\n"
                    "[dim]Please specify the repository name:[/dim]\n"
                    "  refactron repo connect <repo-name>\n"
                )
                raise SystemExit(1)
    else:
        # No path provided - must have repo_name for cloning
        if not repo_name:
            console.print(
                "[red]Error: Repository name is required when not in a git directory.[/red]\n\n"
                "[dim]Usage:[/dim]\n"
                "  refactron repo connect <repo-name>    # Auto-clone to workspace\n"
                "  refactron repo connect --path .       # Link current directory\n"
            )
            raise SystemExit(1)

    # Fetch available repositories
    try:
        with console.status("[primary]Fetching repositories...[/primary]"):
            repositories = list_repositories(api_base_url)
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    # Find matching repository
    matching_repo: Optional[Repository] = None
    for repository in repositories:
        if (
            repository.name.lower() == repo_name.lower()
            or repository.full_name.lower() == repo_name.lower()
        ):
            matching_repo = repository
            break

    if not matching_repo:
        console.print(
            f"[red]Repository '{repo_name}' not found in your connected repositories.[/red]\n"
        )
        console.print("[dim]Available repositories:[/dim]")
        for repository in repositories[:5]:
            console.print(f"  - {repository.full_name}")
        if len(repositories) > 5:
            console.print(f"  ... and {len(repositories) - 5} more")
        console.print("\n[dim]Run 'refactron repo list' to see all repositories.[/dim]")
        raise SystemExit(1)

    # If no path provided, clone to managed workspace
    if not path:
        workspace_root = Path.home() / ".refactron" / "workspaces"
        workspace_root.mkdir(parents=True, exist_ok=True)
        local_path = workspace_root / matching_repo.name

        # Check if already cloned
        if local_path.exists():
            console.print(f"[dim]Repository already exists at: {local_path}[/dim]\n")
        else:
            # Clone the repository
            console.print(f"[primary]Cloning {matching_repo.full_name}...[/primary]\n")

            try:
                subprocess.run(
                    ["git", "clone", matching_repo.clone_url, str(local_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                console.print(f"[success]✓ Cloned successfully to {local_path}[/success]\n")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to clone repository:[/red]\n{e.stderr}")
                raise SystemExit(1)
            except FileNotFoundError:
                console.print(
                    "[red]Error: git command not found.[/red]\n"
                    "[dim]Please install git or use --path to connect an existing directory.[/dim]"
                )
                raise SystemExit(1)

    # Create workspace mapping
    mapping = WorkspaceMapping(
        repo_id=matching_repo.id,
        repo_name=matching_repo.name,
        repo_full_name=matching_repo.full_name,
        local_path=str(local_path),
        connected_at=datetime.now(timezone.utc).isoformat(),
    )

    workspace_mgr.add_workspace(mapping)

    # Trigger background indexing via subprocess
    # We spawn a separate process so it survives after this CLI command exits
    console.print("[dim]Spawning background indexer...[/dim]")
    try:
        # Run 'refactron rag index' in the background
        # We redirect output to DEVNULL to keep it quiet
        pid = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "refactron.cli",
                "rag",
                "index",
                "--background",
            ],
            cwd=str(local_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach from terminal
        ).pid
        console.print(f"[dim]Indexing started in background (PID: {pid}).[/dim]")
        console.print("[dim]Run 'refactron rag status' to check progress.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Auto-indexing failed to start: {e}[/yellow]")

    # Create helpful navigation command
    cd_command = f"cd {local_path}"

    console.print(
        Panel(
            f"[success]Successfully connected![/success]\n\n"
            f"Repository: [bold]{matching_repo.full_name}[/bold]\n"
            f"Local Path: [bold]{local_path}[/bold]\n\n"
            f"[yellow]To navigate to this directory, run:[/yellow]\n"
            f"[bold cyan]{cd_command}[/bold cyan]",
            title="✓ Connected",
            border_style="success",
            box=box.ROUNDED,
        )
    )

    # Also print the cd command separately for easy copying
    console.print(f"\n[dim]Quick copy:[/dim] [bold cyan]{cd_command}[/bold cyan]\n")


@repo.command("disconnect")
@click.argument("repo_name", required=False)
@click.option(
    "--delete-files",
    is_flag=True,
    help="Also delete the local directory (requires confirmation)",
)
def repo_disconnect(repo_name: Optional[str], delete_files: bool) -> None:
    """
    Disconnect a repository and optionally delete local files.

    REPO_NAME: Name of the repository to disconnect (e.g., 'volumeofsphere' or 'user/volumeofsphere')  # noqa: E501

    If not provided, attempts to detect from current directory.
    """
    _setup_logging()
    console.print()
    _auth_banner("Disconnect Repository")
    console.print()

    workspace_mgr = WorkspaceManager()

    # Auto-detect if not provided
    if not repo_name:
        current_workspace = workspace_mgr.get_workspace_by_path(str(Path.cwd()))
        if current_workspace:
            repo_name = current_workspace.repo_full_name
            console.print(f"[dim]Detected repository: {repo_name}[/dim]\n")
        else:
            console.print(
                "[red]Error: No repository specified and current directory is not a connected workspace.[/red]\n\n"  # noqa: E501
                "[dim]Usage:[/dim]\n"
                "  refactron repo disconnect <repo-name>\n"
                "  cd <workspace-dir> && refactron repo disconnect\n"
            )
            raise SystemExit(1)

    # Find the workspace
    workspace = workspace_mgr.get_workspace(repo_name)
    if not workspace:
        console.print(f"[yellow]Repository '{repo_name}' is not connected.[/yellow]\n")
        console.print("[dim]Run 'refactron repo list' to see connected repositories.[/dim]")
        raise SystemExit(1)

    local_path = Path(workspace.local_path)

    # Confirm deletion if requested
    if delete_files:
        if not local_path.exists():
            console.print(f"[yellow]Local directory does not exist: {local_path}[/yellow]\n")
        else:
            console.print(
                Panel(
                    f"[yellow]⚠️  WARNING: This will permanently delete:[/yellow]\n\n"
                    f"[bold]{local_path}[/bold]\n\n"
                    f"[dim]This action cannot be undone![/dim]",
                    title="Confirm Deletion",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )

            if not click.confirm(
                "\nAre you sure you want to delete this directory?",
                default=False,
            ):
                console.print("[yellow]Deletion cancelled.[/yellow]")
                delete_files = False

    # Remove workspace mapping
    workspace_mgr.remove_workspace(repo_name)
    console.print(f"[success]✓ Removed workspace mapping for '{repo_name}'[/success]\n")

    # Delete files if confirmed
    files_deleted = False
    if delete_files and local_path.exists():
        try:
            import shutil

            shutil.rmtree(local_path)
            console.print(f"[success]✓ Deleted directory: {local_path}[/success]\n")
            files_deleted = True
        except Exception as e:
            console.print(f"[red]Failed to delete directory: {e}[/red]\n")
            raise SystemExit(1)

    # Show appropriate summary
    if not local_path.exists() and not files_deleted:
        # Directory was already gone
        console.print(
            Panel(
                f"[yellow]Workspace mapping removed[/yellow]\n\n"
                f"Repository: [bold]{repo_name}[/bold]\n"
                f"Status: [dim]Local directory was already deleted[/dim]",
                title="✓ Cleaned Up",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
    else:
        # Normal disconnect
        console.print(
            Panel(
                f"[success]Repository disconnected successfully![/success]\n\n"
                f"Repository: [bold]{repo_name}[/bold]\n"
                f"Mapping removed: [bold]Yes[/bold]\n"
                f"Files deleted: [bold]{'Yes' if files_deleted else 'No'}[/bold]",
                title="✓ Disconnected",
                border_style="success",
                box=box.ROUNDED,
            )
        )
