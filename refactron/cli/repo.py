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
@click.argument("repo_or_dir", required=False)
@click.option(
    "--path",
    "-p",
    type=click.Path(file_okay=False),
    default=None,
    help="Local path to connect (deprecated, use 'refactron connect .')",
)
@click.option(
    "--api-base-url",
    default=DEFAULT_API_BASE_URL,
    show_default=True,
    help="Refactron API base URL",
)
@click.option(
    "--ssh",
    is_flag=True,
    help="Clone using SSH instead of HTTPS when falling back to the API",
)
def repo_connect(
    repo_or_dir: Optional[str], path: Optional[str], api_base_url: str, ssh: bool
) -> None:
    """
    Connect to a GitHub repository.

    Run inside a git repository to connect it instantly:
      $ refactron repo connect

    Or provide a repository name to clone it from GitHub:
      $ refactron repo connect user/my-project

    Or clone it via SSH instead of HTTPS:
      $ refactron repo connect --ssh user/my-project
    """
    _setup_logging()
    console.print()
    _auth_banner("Connect Repository")
    console.print()

    workspace_mgr = WorkspaceManager()

    # Determine the target local path
    # Priority: 1. --path (deprecated), 2. repo_or_dir if it's a directory,
    # 3. current working directory
    local_path = Path.cwd()
    if path:
        local_path = Path(path).resolve()
    elif repo_or_dir and Path(repo_or_dir).is_dir():
        local_path = Path(repo_or_dir).resolve()

    # == SOURCE A: Local Offline Flow (Primary) ==
    console.print("[dim]Checking for local git repository...[/dim]")
    detected_repo = workspace_mgr.detect_repository(local_path)

    if detected_repo:
        # We are inside a valid git repository! Connect offline.
        repo_name = detected_repo.split("/")[-1] if "/" in detected_repo else detected_repo

        mapping = WorkspaceMapping(
            repo_id=None,  # Offline path doesn't know the DB ID
            repo_name=repo_name,
            repo_full_name=detected_repo,
            local_path=str(local_path),
            connected_at=datetime.now(timezone.utc).isoformat(),
        )
        workspace_mgr.add_workspace(mapping)

        # Trigger background indexing
        _spawn_background_indexer(local_path)

        console.print(
            Panel(
                f"[success]Successfully connected![/success]\n\n"
                f"Repository: [bold]{detected_repo}[/bold] [dim](detected from .git/config)[/dim]\n"
                f"Workspace:  [bold]{local_path}[/bold]\n\n"
                f"[yellow]Next step:[/yellow]\n"
                f"[bold cyan]refactron analyze .[/bold cyan]",
                title="✓ Connected (Local)",
                border_style="success",
                box=box.ROUNDED,
            )
        )
        return

    # == SOURCE B: API Fallback Flow ==
    # We only get here if there is no local git repository found.

    # We must have a repo name to clone if we are not in a git repo.
    if not repo_or_dir or Path(repo_or_dir).is_dir():
        console.print(
            "[red]Error: Not a git repository. Run inside a git repo or "
            "provide a repo name.[/red]\n\n"
            "[dim]Usage:[/dim]\n"
            "  refactron repo connect                # Auto-detect current directory\n"
            "  refactron repo connect <repo-name>    # Auto-clone from GitHub\n"
        )
        raise SystemExit(1)

    repo_name_target = repo_or_dir

    try:
        with console.status("[primary]Fetching repositories from API...[/primary]"):
            repositories = list_repositories(api_base_url)
    except RuntimeError as e:
        console.print(f"[red]Authentication required for cloning:[/red] {e}")
        console.print(
            "[dim]Run 'refactron repo connect' inside an existing git "
            "repository to connect offline.[/dim]"
        )
        raise SystemExit(1)

    matching_repo: Optional[Repository] = None
    for repository in repositories:
        if (
            repository.name.lower() == repo_name_target.lower()
            or repository.full_name.lower() == repo_name_target.lower()
        ):
            matching_repo = repository
            break

    if not matching_repo:
        console.print(
            f"[red]Repository '{repo_name_target}' not found in your "
            "connected repositories.[/red]\n"
        )
        console.print("[dim]Available repositories:[/dim]")
        for repository in repositories[:5]:
            console.print(f"  - {repository.full_name}")
        if len(repositories) > 5:
            console.print(f"  ... and {len(repositories) - 5} more")
        raise SystemExit(1)

    # Pyre2 needs a hint that matching_repo is definitely not None here
    if matching_repo is None:
        raise SystemExit(1)

    # Clone to managed workspace if we didn't explicitly ask to map a local un-git path
    workspace_root = Path.home() / ".refactron" / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    clone_path = workspace_root / matching_repo.name

    if clone_path.exists():
        console.print(f"[dim]Repository already exists at: {clone_path}[/dim]\n")
    else:
        clone_target_url = matching_repo.ssh_url if ssh else matching_repo.clone_url
        console.print(
            f"[primary]Cloning {matching_repo.full_name} via "
            f"{'SSH' if ssh else 'HTTPS'}...[/primary]\n"
        )
        try:
            subprocess.run(
                ["git", "clone", clone_target_url, str(clone_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            console.print(f"[success]✓ Cloned successfully to {clone_path}[/success]\n")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to clone repository:[/red]\n{e.stderr}")
            if not ssh:
                console.print(
                    "\n[yellow]Hint: If HTTPS cloning is blocked or failing, "
                    "try using SSH instead:[/yellow]\n"
                    f"[bold cyan]  refactron repo connect --ssh {repo_name_target}[/bold cyan]"
                )
            raise SystemExit(1)
        except FileNotFoundError:
            console.print("[red]Error: git command not found.[/red]")
            raise SystemExit(1)

    mapping = WorkspaceMapping(
        repo_id=matching_repo.id,
        repo_name=matching_repo.name,
        repo_full_name=matching_repo.full_name,
        local_path=str(clone_path),
        connected_at=datetime.now(timezone.utc).isoformat(),
    )
    workspace_mgr.add_workspace(mapping)

    _spawn_background_indexer(clone_path)

    cd_command = f"cd {clone_path}"
    console.print(
        Panel(
            f"[success]Successfully connected![/success]\n\n"
            f"Repository: [bold]{matching_repo.full_name}[/bold] [dim](cloned from GitHub)[/dim]\n"
            f"Workspace:  [bold]{clone_path}[/bold]\n\n"
            f"[yellow]To navigate to this directory, run:[/yellow]\n"
            f"[bold cyan]{cd_command}[/bold cyan]\n\n"
            f"[yellow]Next step:[/yellow]\n"
            f"[bold cyan]refactron analyze .[/bold cyan]",
            title="✓ Connected (API)",
            border_style="success",
            box=box.ROUNDED,
        )
    )


def _spawn_background_indexer(local_path: Path) -> None:
    """Helper to start the RAG indexer in the background."""
    console.print("[dim]Spawning background indexer...[/dim]")
    try:
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
            start_new_session=True,
        ).pid
        console.print(f"[dim]Indexing started in background (PID: {pid}).[/dim]")
        console.print("[dim]Run 'refactron rag status' to check progress.[/dim]")
    except Exception as e:
        console.print(f"[yellow]Auto-indexing failed to start: {e}[/yellow]")


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
