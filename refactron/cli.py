"""Command-line interface for Refactron."""

import logging
import platform
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import click
import requests  # type: ignore
import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from refactron import Refactron, __version__
from refactron.autofix.engine import AutoFixEngine
from refactron.autofix.models import FixRiskLevel
from refactron.core.analysis_result import AnalysisResult
from refactron.core.backup import BackupRollbackSystem
from refactron.core.config import RefactronConfig
from refactron.core.credentials import (
    RefactronCredentials,
    credentials_path,
    delete_credentials,
    load_credentials,
    save_credentials,
)
from refactron.core.device_auth import (
    DEFAULT_API_BASE_URL,
    poll_for_token,
    start_device_authorization,
)
from refactron.core.exceptions import ConfigError
from refactron.core.refactor_result import RefactorResult
from refactron.patterns.storage import PatternStorage
from refactron.patterns.tuner import RuleTuner

# Custom theme for a premium, modern look
THEME = Theme(
    {
        "primary": "bold #5f5fff",  # A vibrant indigo/blue
        "secondary": "#8a8a8a",  # Sleek gray
        "success": "bold #00d787",  # Modern mint/green
        "warning": "bold #ffaf00",  # Warm amber
        "error": "bold #ff5f5f",  # Soft red
        "info": "#5fafff",  # Sky blue
        "highlight": "bold #ffffff",  # Bright white
        "link": "underline #5f5fff",  # Link color
        "panel.border": "#444444",  # Subtle border
    }
)

console = Console(theme=THEME)


@dataclass(frozen=True)
class ApiKeyValidationResult:
    ok: bool
    message: str


def _auth_banner(title: str) -> None:
    """Display a premium, stylized banner."""
    # Create a modern header with a tagline
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)

    # Title with gradient-like effect (simulated with colors)
    header_text = Text()
    header_text.append("Refactron", style="primary")
    header_text.append(" | ", style="secondary")
    header_text.append(title, style="highlight")

    grid.add_row(header_text)
    grid.add_row(Text("The Intelligent Code Refactoring Transformer", style="secondary italic"))

    console.print(
        Panel(
            grid,
            style="panel.border",
            box=box.ROUNDED,
            padding=(1, 2),
            subtitle="[secondary]v1.0.14[/secondary]",
            subtitle_align="right",
        )
    )


def _validate_api_key(
    api_base_url: str, api_key: str, timeout_seconds: int
) -> ApiKeyValidationResult:
    """
    Validate an API key against the backend before saving it locally.

    The key is sent as a Bearer token to a small verification endpoint. We keep
    the UX actionable: distinguish invalid keys from missing endpoints and
    connectivity issues.
    """
    url = f"{api_base_url.rstrip('/')}/api/auth/verify-key"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
    except requests.Timeout:
        return ApiKeyValidationResult(
            ok=False,
            message="API key verification timed out. Is the API reachable?",
        )
    except requests.ConnectionError:
        return ApiKeyValidationResult(
            ok=False,
            message="Could not reach the Refactron API. Is it running?",
        )
    except requests.RequestException:
        return ApiKeyValidationResult(
            ok=False,
            message="API key verification failed due to a network error.",
        )

    if response.status_code == 200:
        return ApiKeyValidationResult(ok=True, message="Verified.")
    if response.status_code in (401, 403):
        return ApiKeyValidationResult(ok=False, message="Invalid API key.")
    if response.status_code == 404:
        return ApiKeyValidationResult(
            ok=False,
            message="API key verification endpoint is missing (404).",
        )
    if 500 <= response.status_code <= 599:
        return ApiKeyValidationResult(
            ok=False,
            message=f"API key verification failed (server error {response.status_code}).",
        )
    return ApiKeyValidationResult(
        ok=False,
        message=f"API key verification failed (HTTP {response.status_code}).",
    )


def _setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(
    config_path: Optional[str],
    profile: Optional[str] = None,
    environment: Optional[str] = None,
) -> RefactronConfig:
    """Load configuration from file or use default."""
    try:
        if config_path:
            console.print(f"[dim]Loading config from: {config_path}[/dim]")
            if profile or environment:
                env_display = environment or profile
                console.print(f"[dim]Using profile/environment: {env_display}[/dim]")
            return RefactronConfig.from_file(Path(config_path), profile, environment)
        return RefactronConfig.default()
    except ConfigError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        if e.recovery_suggestion:
            console.print(f"[yellow]Tip: {e.recovery_suggestion}[/yellow]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error loading configuration: {e}[/red]")
        raise SystemExit(1)


def _validate_path(target: str) -> Path:
    """Validate target path exists."""
    target_path = Path(target)
    if not target_path.exists():
        console.print(f"[red]Error: Path does not exist: {target}[/red]")
        raise SystemExit(1)
    return target_path


def _print_file_count(target_path: Path) -> None:
    """Print count of Python files if target is directory."""
    if target_path.is_dir():
        py_files = list(target_path.rglob("*.py"))
        console.print(f"[dim]Found {len(py_files)} Python file(s) to analyze[/dim]\n")


def _create_summary_table(summary: dict) -> Table:
    """Create analysis summary table."""
    table = Table(
        title="Analysis Summary",
        show_header=True,
        header_style="primary",
        box=box.ROUNDED,
        border_style="panel.border",
        expand=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold white")

    table.add_row("Files Found", str(summary["total_files"]))
    table.add_row("Files Analyzed", str(summary["files_analyzed"]))
    if summary.get("files_failed", 0) > 0:
        table.add_row("Files Failed", str(summary["files_failed"]), style="bold red")

    table.add_row(
        "Total Issues",
        str(summary["total_issues"]),
        style="bold yellow" if summary["total_issues"] > 0 else "bold green",
    )

    if summary["critical"] > 0:
        table.add_row("Critical", str(summary["critical"]), style="bold red")
    else:
        table.add_row("Critical", "0", style="dim")

    table.add_row(
        "Errors", str(summary["errors"]), style="bold red" if summary["errors"] > 0 else "dim"
    )
    table.add_row(
        "Warnings",
        str(summary["warnings"]),
        style="bold yellow" if summary["warnings"] > 0 else "dim",
    )
    table.add_row("Info", str(summary["info"]), style="cyan" if summary["info"] > 0 else "dim")

    return table


def _detect_project_type() -> Optional[str]:
    """
    Detect project type by checking for framework-specific files and imports.

    Detection patterns:
    - Django: Checks for settings.py or manage.py files with Django-specific imports/variables
    - FastAPI: Looks for 'from fastapi import' or 'import fastapi' in common entry points
    - Flask: Looks for 'from flask import' with Flask app instantiation patterns

    Returns:
        Detected framework name ('django', 'fastapi', 'flask') or None
    """
    current_dir = Path.cwd()

    # Check for Django first (manage.py or settings.py are strong signals)
    for django_file in ["manage.py", "**/settings.py"]:
        for file_path in current_dir.glob(django_file):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read line by line to avoid loading large files
                    for line in f:
                        if "django" in line.lower() or "DJANGO_SETTINGS_MODULE" in line:
                            return "django"
            except (IOError, OSError):
                pass

    # Check common entry point files for FastAPI and Flask
    common_entry_points = ["main.py", "app.py", "application.py", "server.py", "api.py"]
    for entry_point in common_entry_points:
        for file_path in current_dir.glob(entry_point):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Check for FastAPI
                    if "from fastapi import" in content or "import fastapi" in content:
                        return "fastapi"

                    # Check for Flask
                    if "from flask import" in content or "import flask" in content:
                        if "Flask(__name__)" in content or "app = Flask" in content:
                            return "flask"
            except (IOError, OSError):
                pass

    return None


def _print_status_messages(summary: dict) -> None:
    """Print status messages based on analysis results."""
    if summary.get("files_failed", 0) > 0:
        console.print(
            f"[warning]{summary['files_failed']} file(s) failed analysis "
            f"and were skipped[/warning]"
        )

    if summary["total_issues"] == 0 and summary.get("files_failed", 0) == 0:
        console.print(
            Panel(
                "[success]Excellent! No issues found.[/success]",
                box=box.ROUNDED,
                border_style="success",
            )
        )
    elif summary["total_issues"] == 0 and summary.get("files_failed", 0) > 0:
        console.print(
            "[warning]No issues found in analyzed files, but some files failed.[/warning]"
        )
    elif summary["critical"] > 0:
        console.print(
            f"[error]Found {summary['critical']} critical issue(s) that need immediate "
            f"attention![/error]"
        )


def _print_detailed_issues(result: AnalysisResult) -> None:
    """Print detailed issues list."""
    console.print("[primary bold]Detailed Issues:[/primary bold]\n")

    for issue in result.all_issues:
        style = (
            "error"
            if issue.level.value in ("critical", "error")
            else "warning" if issue.level.value == "warning" else "info"
        )
        level_label = f"[{issue.level.value.upper()}]"

        console.print(f"[{style}]{level_label} {issue}[/{style}]")
        if issue.suggestion:
            console.print(f"   [secondary]Tip: {issue.suggestion}[/secondary]")
        console.print()


def _print_helpful_tips(summary: dict, detailed: bool) -> None:
    """Print helpful tips based on results."""
    if summary["total_issues"] > 0 and not detailed:
        console.print("[secondary]Tip: Use --detailed to see all issues[/secondary]")

    if summary["total_issues"] > 5:
        console.print(
            "[secondary]Tip: Run 'refactron refactor --preview' to see suggested fixes[/secondary]"
        )


def _print_refactor_filters(types: tuple) -> None:
    """Print operation type filters if specified."""
    if types:
        console.print(f"[secondary]Filtering for: {', '.join(types)}[/secondary]\n")


def _confirm_apply_mode(preview: bool) -> None:
    """Warn and confirm if using --apply mode."""
    if not preview:
        console.print("[warning]--apply mode will modify your files![/warning]")
        if not click.confirm("Continue?"):
            raise SystemExit(0)


def _create_refactor_table(summary: dict) -> Table:
    """Create refactoring summary table."""
    table = Table(
        title="Refactoring Summary",
        show_header=True,
        header_style="primary",
        box=box.ROUNDED,
        border_style="panel.border",
        expand=True,
    )
    table.add_column("Metric", style="info")
    table.add_column("Value", justify="right", style="highlight")

    table.add_row("Total Operations", str(summary["total_operations"]))
    table.add_row("Safe Operations", str(summary["safe"]), style="success")

    if summary["high_risk"] > 0:
        table.add_row("High Risk", str(summary["high_risk"]), style="error bold")
    else:
        table.add_row("High Risk", "0", style="secondary")

    table.add_row("Applied", "Yes" if summary["applied"] else "No", style="highlight")

    return table


def _print_refactor_messages(summary: dict, preview: bool) -> None:
    """Print status messages for refactoring results."""
    if summary["total_operations"] == 0:
        console.print(
            Panel(
                "[success]No refactoring opportunities found. Your code looks good![/success]",
                box=box.ROUNDED,
                border_style="success",
            )
        )
    elif summary["high_risk"] > 0:
        console.print(
            f"[warning]{summary['high_risk']} operation(s) are high-risk. Review "
            f"carefully![/warning]"
        )

    if preview and summary["total_operations"] > 0:
        console.print("\n[info]This is a preview. Use --apply to apply changes.[/info]")
        console.print("[secondary]Tip: Review each change carefully before applying[/secondary]")

    if summary["total_operations"] > 0 and summary["applied"]:
        console.print("\n[success]Refactoring completed! Don't forget to test your code.[/success]")


def _collect_feedback_interactive(refactron: Refactron, result: RefactorResult) -> None:
    """
    Collect feedback from user interactively for each refactoring operation.

    Args:
        refactron: Refactron instance to record feedback
        result: RefactorResult containing operations
    """
    if not result.operations:
        return

    console.print("\n[bold]Feedback Collection (Optional)[/bold]")
    console.print("[dim]Help us learn from your feedback to improve suggestions![/dim]\n")

    for op in result.operations:
        console.print(f"\n[cyan]Operation ID: {op.operation_id}[/cyan]")
        console.print(f"[dim]Type: {op.operation_type} at {op.file_path}:{op.line_number}[/dim]")
        ranking_score = result.get_ranking_score(op)
        if ranking_score > 0:
            console.print(f"[dim]Ranking Score: {ranking_score:.3f}[/dim]")

        action = click.prompt(
            "Your feedback? (a)ccepted, (r)ejected, (i)gnored, or (s)kip",
            type=click.Choice(["a", "r", "i", "s"], case_sensitive=False),
            default="s",
        ).lower()

        if action == "s":
            continue

        reason = None
        if action in ("r", "a", "i"):
            reason = click.prompt(
                "Reason (optional, press Enter to skip)",
                default="",
                show_default=False,
            )
            if not reason.strip():
                reason = None

        action_map = {"a": "accepted", "r": "rejected", "i": "ignored"}
        refactron.record_feedback(
            operation_id=op.operation_id,
            action=action_map[action],
            reason=reason,
            operation=op,
        )

    console.print("\n[success]Thank you for your feedback![/success]")


def _record_applied_operations(refactron: Refactron, result: RefactorResult) -> None:
    """
    Automatically record all operations as accepted when --apply is used.

    Args:
        refactron: Refactron instance to record feedback
        result: RefactorResult containing operations
    """
    if not result.operations:
        return

    for op in result.operations:
        refactron.record_feedback(
            operation_id=op.operation_id,
            action="accepted",
            reason="Applied via --apply flag",
            operation=op,
        )


def _run_startup_animation() -> None:
    """Run a sleek startup animation with a big logo and system info."""
    import random
    import time

    from rich.align import Align
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    # Clear screen first
    console.clear()

    LOGO_LINES = [
        r"██████╗ ███████╗███████╗ █████╗  ██████╗████████╗██████╗  ██████╗ ███╗   ██╗",
        r"██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║",
        r"██████╔╝█████╗  █████╗  ███████║██║        ██║   ██████╔╝██║   ██║██╔██╗ ██║",
        r"██╔══██╗██╔══╝  ██╔══╝  ██╔══██║██║        ██║   ██╔══██╗██║   ██║██║╚██╗██║",
        r"██║  ██║███████╗██║     ██║  ██║╚██████╗   ██║   ██║  ██║╚██████╔╝██║ ╚████║",
        r"╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
    ]

    subtitle_text = "The Intelligent Code Refactoring Transformer"

    TIPS = [
        "Tip: Use 'refactron analyze' to find technical debt.",
        "Tip: 'refactron rollback' can undo your last changes.",
        "Tip: Check '.refactron.yaml' to customize behavior.",
        "Tip: Run 'refactron serve-metrics' for Prometheus data.",
        "Tip: Use --detailed for a deeper analysis report.",
    ]

    CHECKS = [
        "Verifying configuration...",
        "Checking credentials...",
        "Initializing refactoring engine...",
        "Scanning project structure...",
        "Ready to transform.",
    ]

    selected_tip = random.choice(TIPS)

    def get_renderable(step: int, phase: str) -> Align:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(Text("\n" * 2))  # Top padding

        if phase == "check":
            # System check phase
            check_idx = min(step // 5, len(CHECKS) - 1)
            grid.add_row(Align.center(Text("SYSTEM CHECK", style="bold #5f5fff")))
            grid.add_row(Align.center(Text(CHECKS[check_idx], style="dim")))

            # Mini progress bar
            width = 30
            filled = int((step / 25) * width)
            bar = "━" * filled + " " * (width - filled)
            grid.add_row(Align.center(Text(f"[{bar}]", style="#5f5fff")))

        elif phase == "logo":
            # Reveal logo wipe
            max_len = max(len(line) for line in LOGO_LINES)
            reveal_len = int((step / 30) * max_len)

            logo_text = Text()
            for line in LOGO_LINES:
                visible_part = line[:reveal_len]
                logo_text.append(visible_part + "\n", style="bold #ffffff")

            grid.add_row(Align.center(logo_text))

        elif phase == "final":
            # Logo static
            logo_text = Text()
            for line in LOGO_LINES:
                logo_text.append(line + "\n", style="bold #ffffff")
            grid.add_row(Align.center(logo_text))

            # Subtitle
            grid.add_row(Align.center(Text(subtitle_text, style="italic #8a8a8a")))

            # System Info
            info_table = Table.grid(padding=(0, 2))
            info_table.add_column(style="dim", justify="right")
            info_table.add_column(style="bold white")

            info_table.add_row("Version:", "v1.0.13")
            info_table.add_row("Python:", sys.version.split()[0])
            info_table.add_row("OS:", platform.system())

            grid.add_row(Text("\n"))
            grid.add_row(Align.center(info_table))

            # Tip
            grid.add_row(Text("\n"))
            grid.add_row(
                Align.center(
                    Panel(Text(selected_tip, style="cyan"), border_style="#333333", expand=False)
                )
            )

        return Align.center(grid)

    with Live(console=console, refresh_per_second=20, transient=True) as live:
        # Phase 1: System Checks
        for i in range(26):
            live.update(get_renderable(i, "check"))
            time.sleep(0.04)

        # Phase 2: Logo Wipe
        for i in range(31):
            live.update(get_renderable(i, "logo"))
            time.sleep(0.03)

        # Phase 3: Final Reveal
        live.update(get_renderable(0, "final"))
        time.sleep(1.5)

    # Final static print
    console.print()
    for line in LOGO_LINES:
        console.print(Align.center(Text(line, style="bold #ffffff")))
    console.print(Align.center(Text(subtitle_text, style="italic #8a8a8a")))
    console.print(Align.center(Text("v1.0.13", style="dim")))
    console.print()


def _print_custom_help(ctx: click.Context) -> None:
    """Print a beautifully formatted, numbered help screen."""
    console.print()
    # Use a clean, bold header
    header = Table.grid(expand=True)
    header.add_column(justify="center")
    header.add_row(Text("⚡ REFACTRON", style="bold white"))
    header.add_row(Text("INTELLIGENT CODE REFACTORING", style="dim white"))
    console.print(Panel(header, border_style="#333333", padding=(1, 2)))

    console.print("\n[bold white]COMMAND CENTER[/bold white]")
    console.print("[dim]Select a command by name or number[/dim]\n")

    table = Table(
        show_header=True, header_style="bold cyan", box=box.SIMPLE, expand=True, padding=(0, 2)
    )
    table.add_column("ID", justify="right", style="cyan", width=4)
    table.add_column("COMMAND", style="bold white", width=20)
    table.add_column("DESCRIPTION", style="dim")

    commands = sorted(ctx.command.list_commands(ctx))
    for i, cmd_name in enumerate(commands, 1):
        cmd = ctx.command.get_command(ctx, cmd_name)
        description = cmd.get_short_help_str() if cmd else ""
        table.add_row(f"{i:02d}", cmd_name.upper(), description)

    console.print(table)

    console.print("\n[bold cyan]GLOBAL OPTIONS[/bold cyan]")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold white")
    grid.add_column(style="dim")
    grid.add_row("--version", "Show the version and exit.")
    grid.add_row("--help", "Show this message and exit.")
    console.print(grid)

    console.print("\n[dim]USAGE: refactron <command> [args]...[/dim]")
    console.print("[dim]EXAMPLE: refactron analyze . --detailed[/dim]\n")


class CustomHelpGroup(click.Group):
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _print_custom_help(ctx)


def _run_minimal_loop(ctx: click.Context) -> None:
    """Run a minimal interactive loop for help and version only."""

    def print_header() -> None:
        creds = load_credentials()
        user = creds.email if creds else "Guest"
        plan = (creds.plan or "Free").upper() if creds else "N/A"

        # Profile Card Layout
        grid = Table.grid(expand=False, padding=(0, 3))
        grid.add_column(justify="center")
        grid.add_column(justify="left", vertical="middle")

        # Avatar (Simple ASCII)
        avatar = """
      ▄▄▄
     █████
    ███████
   █████████
  ███ █ █ ███
        """

        info = Table.grid(padding=(0, 1))
        info.add_column(style="dim", justify="right")
        info.add_column(style="bold white")

        info.add_row("User:", user)
        info.add_row("Plan:", plan)
        info.add_row("Status:", "[green]Online[/green]")

        grid.add_row(Text(avatar.strip(), style="#5f5fff"), info)

        console.print(
            Panel(
                grid,
                title="[bold]DASHBOARD[/bold]",
                border_style="#444444",
                box=box.ROUNDED,
                expand=False,
                padding=(1, 2),
            )
        )

    console.clear()
    print_header()

    while True:
        try:
            console.print("\n[bold]Available Options:[/bold]")
            console.print(
                "1. [bold blue]Help[/bold blue]      [dim]Show CLI usage and commands[/dim]"
            )
            console.print("2. [bold blue]Version[/bold blue]   [dim]Show current version[/dim]")
            console.print("3. [bold blue]Exit[/bold blue]")

            choice = Prompt.ask(
                "\n[bold]>[/bold] ", choices=["1", "2", "3"], default="3", show_choices=False
            )

            if choice == "1":
                _print_custom_help(ctx)
            elif choice == "2":
                console.print("\nRefactron CLI v1.0.13")
            elif choice == "3":
                console.print("Goodbye!")
                break

        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break


@click.group(cls=CustomHelpGroup, invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def main(ctx: click.Context) -> None:
    """
    Refactron - The Intelligent Code Refactoring Transformer

    Analyze, refactor, and optimize your Python code with ease.
    """
    # Check authentication for all commands except login/logout
    exempt_commands = ["login", "logout", "auth"]

    # 1. Pre-check authentication status
    creds = load_credentials()
    is_authenticated = False
    if creds and creds.access_token:
        now = datetime.now(timezone.utc)
        if not creds.expires_at or creds.expires_at > now:
            is_authenticated = True

    # 2. Show animation if dashboard mode OR if auth is required and missing
    should_show_animation = ctx.invoked_subcommand is None or (
        ctx.invoked_subcommand not in exempt_commands and not is_authenticated
    )

    if should_show_animation:
        _run_startup_animation()

    # 3. Handle authentication requirement
    if ctx.invoked_subcommand not in exempt_commands and not is_authenticated:
        # If it's a subcommand, we might want a slightly different message
        if ctx.invoked_subcommand:
            console.print(
                f"\n[yellow]Authentication required to run '{ctx.invoked_subcommand}'[/yellow]"
            )
        else:
            console.print(Align.center(Text("\nAuthentication Required", style="bold")))

        if Prompt.ask("\nLog in to continue?", choices=["y", "n"], default="y") == "y":
            try:
                ctx.invoke(
                    login,
                    api_base_url=DEFAULT_API_BASE_URL,
                    no_browser=False,
                    timeout=300,
                    force=False,
                )
                # Re-check credentials
                creds = load_credentials()
                if creds and creds.access_token:
                    is_authenticated = True
            except SystemExit:
                pass

        if not is_authenticated:
            console.print("[dim]Exiting...[/dim]")
            raise SystemExit(1)

    # 4. Handle default command (interactive dashboard)
    if ctx.invoked_subcommand is None:
        _run_minimal_loop(ctx)
    pass


@main.command()
@click.option(
    "--api-base-url",
    default=DEFAULT_API_BASE_URL,
    show_default=True,
    help="Refactron API base URL",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Do not open a browser automatically (print the URL instead)",
)
@click.option(
    "--timeout",
    type=int,
    default=10,
    show_default=True,
    help="HTTP timeout in seconds for each request",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-login even if already logged in",
)
def login(api_base_url: str, no_browser: bool, timeout: int, force: bool) -> None:
    """Log in to Refactron CLI via device-code flow."""
    import time

    _setup_logging()

    if not force:
        existing = load_credentials()
        if existing and existing.access_token:
            now = datetime.now(timezone.utc)
            if not existing.expires_at or existing.expires_at > now:
                console.print("\n[bold green]Already authenticated[/bold green]")
                console.print(f"User: [dim]{existing.email or 'unknown'}[/dim]")
                return

    with console.status("[bold blue]Connecting to Refactron...[/bold blue]", spinner="dots"):
        time.sleep(0.5)
        try:
            auth = start_device_authorization(api_base_url=api_base_url, timeout_seconds=timeout)
        except Exception as e:
            console.print(
                Panel(f"Failed to start login: {e}", title="Connection Error", border_style="red")
            )
            raise SystemExit(1)

    login_url = f"https://app.refactron.dev/login?{urlencode({'code': auth.user_code})}"

    instructions = Text()
    instructions.append("Please visit the following URL to authenticate:\n\n", style="dim")
    instructions.append(f"  {login_url}\n\n", style="underline bold #5f5fff")
    instructions.append("Verification Code:\n", style="dim")
    instructions.append(f"  {auth.user_code}\n", style="bold white")

    console.print(
        Panel(
            instructions,
            title="Device Login",
            border_style="#444444",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    if not no_browser:
        console.print("[dim]Opening browser...[/dim]")
        try:
            webbrowser.open(login_url, new=2)
        except Exception as e:
            console.print(
                Panel(
                    f"Could not open your browser automatically: {e}\n"
                    "Please open the above URL manually in your browser.",
                    title="Browser Warning",
                    border_style="yellow",
                )
            )

    try:
        with console.status("[bold blue]Waiting for authorization...[/bold blue]", spinner="dots"):
            token = poll_for_token(
                device_code=auth.device_code,
                api_base_url=api_base_url,
                interval_seconds=auth.interval,
                expires_in_seconds=auth.expires_in,
                timeout_seconds=timeout,
            )
    except Exception as e:
        console.print(Panel(f"Login failed: {e}", title="Error", border_style="red"))
        raise SystemExit(1)

    # For pro/enterprise plans, require a verified API key before completing login.
    api_key: Optional[str] = None
    plan_lower = (token.plan or "").lower()
    if plan_lower in ("pro", "enterprise"):
        console.print()
        console.print(
            Panel(
                "Your plan requires an API key.\n\n"
                "Generate a key in the Refactron web app and paste it below.",
                title="API Key Required",
                border_style="#444444",
                box=box.ROUNDED,
            )
        )
        api_key_input = click.prompt("API key", hide_input=True, default="")
        candidate_key = api_key_input.strip()
        if not candidate_key:
            console.print(
                Panel(
                    "API key is required for this plan.", title="Login aborted", border_style="red"
                )
            )
            raise SystemExit(1)

        with console.status("[bold blue]Verifying API key...[/bold blue]", spinner="dots"):
            time.sleep(0.5)
            validation = _validate_api_key(
                api_base_url=api_base_url,
                api_key=candidate_key,
                timeout_seconds=timeout,
            )

        if not validation.ok:
            console.print(
                Panel(
                    f"{validation.message}\n\nAPI: {api_base_url}",
                    title="Login aborted",
                    border_style="red",
                    box=box.ROUNDED,
                )
            )
            raise SystemExit(1)

        api_key = candidate_key
        console.print(Panel("API key verified.", border_style="success", box=box.ROUNDED))

    creds = RefactronCredentials(
        api_base_url=api_base_url,
        access_token=token.access_token,
        token_type=token.token_type,
        expires_at=token.expires_at(),
        email=token.email,
        plan=token.plan,
        api_key=api_key,
    )

    try:
        save_credentials(creds)
    except Exception as e:
        console.print(
            Panel(f"Failed to save credentials: {e}", title="Error", border_style="error")
        )
        raise SystemExit(1)

    expires_at_local = creds.expires_at
    expires_at_str = (
        expires_at_local.astimezone(timezone.utc).isoformat() if expires_at_local else "unknown"
    )

    who = creds.email or "unknown"
    plan = creds.plan or "unknown"

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("k", style="secondary", no_wrap=True)
    summary.add_column("v", style="highlight")
    summary.add_row("User", who)
    summary.add_row("Plan", plan)
    summary.add_row("Token expires", expires_at_str)

    if plan_lower in ("pro", "enterprise"):
        summary.add_row("API key", "Configured" if creds.api_key else "Missing")
    summary.add_row("Credentials file", str(credentials_path()))

    console.print()
    console.print(
        Panel(
            summary, title="Login complete", border_style="success", box=box.ROUNDED, padding=(1, 2)
        )
    )
    console.print()


@main.command()
def logout() -> None:
    """Log out of Refactron CLI (removes stored credentials)."""
    _setup_logging()
    console.print()
    _auth_banner("Logout")
    console.print()
    deleted = delete_credentials()
    if deleted:
        console.print(
            Panel(
                "Stored credentials removed.",
                title="Logged out",
                border_style="success",
                box=box.ROUNDED,
            )
        )
    else:
        console.print(
            Panel(
                "No stored credentials found.",
                title="Logout",
                border_style="warning",
                box=box.ROUNDED,
            )
        )


@main.group()
def auth() -> None:
    """Authentication commands."""
    pass


@auth.command("status")
def auth_status() -> None:
    """Show current login status."""
    _setup_logging()
    console.print()
    _auth_banner("Auth status")
    console.print()
    creds = load_credentials()
    if not creds:
        console.print(
            Panel("Not logged in.", title="Authentication", border_style="warning", box=box.ROUNDED)
        )
        return

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="secondary", no_wrap=True)
    table.add_column("v", style="highlight")
    table.add_row("Status", "Logged in")

    table.add_row("User", creds.email or "unknown")
    table.add_row("Plan", creds.plan or "unknown")
    if creds.expires_at:
        table.add_row("Token expires", creds.expires_at.isoformat())
    table.add_row("API key", "Present" if creds.api_key else "Not set")

    console.print(
        Panel(
            table, title="Authentication", border_style="primary", box=box.ROUNDED, padding=(1, 2)
        )
    )


@auth.command("logout")
def auth_logout() -> None:
    """Alias for `refactron logout`."""
    logout()


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--detailed/--summary",
    default=True,
    help="Show detailed or summary report",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set log level",
)
@click.option(
    "--log-format",
    type=click.Choice(["json", "text"], case_sensitive=False),
    help="Set log format (json for CI/CD, text for console)",
)
@click.option(
    "--metrics/--no-metrics",
    default=None,
    help="Enable or disable metrics collection",
)
@click.option(
    "--show-metrics",
    is_flag=True,
    help="Show metrics summary after analysis",
)
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help=(
        "Named configuration profile to use (dev, staging, prod). "
        "Profiles typically group config defaults; if both --profile and "
        "--environment are set, the environment determines the final "
        "effective configuration."
    ),
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help=(
        "Target runtime environment (dev, staging, prod). When both "
        "--profile and --environment are provided, the environment "
        "overrides the selected profile."
    ),
)
def analyze(
    target: str,
    config: Optional[str],
    detailed: bool,
    log_level: Optional[str],
    log_format: Optional[str],
    metrics: Optional[bool],
    show_metrics: bool,
    profile: Optional[str],
    environment: Optional[str],
) -> None:
    """
    Analyze code for issues and technical debt.

    TARGET: Path to file or directory to analyze
    """
    # Setup logging
    _setup_logging()

    console.print()
    _auth_banner("Analysis")
    console.print()

    # Setup
    target_path = _validate_path(target)
    cfg = _load_config(config, profile, environment)

    # Override config with CLI options
    if log_level:
        cfg.log_level = log_level
    if log_format:
        cfg.log_format = log_format
    if metrics is not None:
        cfg.enable_metrics = metrics

    _print_file_count(target_path)

    # Run analysis
    try:
        with console.status("[primary]Analyzing code...[/primary]"):
            refactron = Refactron(cfg)
            result = refactron.analyze(target)
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        console.print("[dim]Tip: Check if all files have valid Python syntax[/dim]")
        raise SystemExit(1)

    # Display results
    summary = result.summary()
    console.print(_create_summary_table(summary))
    console.print()

    _print_status_messages(summary)

    if detailed and result.all_issues:
        _print_detailed_issues(result)

    _print_helpful_tips(summary, detailed)

    # Show metrics if requested
    if show_metrics and cfg.enable_metrics:
        from refactron.core.metrics import get_metrics_collector

        console.print("\n[bold]Metrics Summary:[/bold]")
        collector = get_metrics_collector()
        metrics_summary = collector.get_analysis_summary()
        console.print(
            f"  Total analysis time: {metrics_summary.get('total_analysis_time_ms', 0):.2f}ms"
        )
        console.print(
            f"  Average time per file: {metrics_summary.get('average_time_per_file_ms', 0):.2f}ms"
        )
        console.print(f"  Success rate: {metrics_summary.get('success_rate_percent', 0):.1f}%")

    # Exit with error code if critical issues found
    if summary["critical"] > 0:
        raise SystemExit(1)


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help=(
        "Named configuration profile to use (dev, staging, prod). "
        "Profiles typically group config defaults; if both --profile and "
        "--environment are set, the environment determines the final "
        "effective configuration."
    ),
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help=(
        "Target runtime environment (dev, staging, prod). When both "
        "--profile and --environment are provided, the environment "
        "overrides the selected profile."
    ),
)
@click.option(
    "--preview/--apply",
    default=True,
    help="Preview changes or apply them",
)
@click.option(
    "--types",
    "-t",
    multiple=True,
    help="Specific refactoring types to apply",
)
@click.option(
    "--feedback/--no-feedback",
    default=False,
    help="Collect interactive feedback on refactoring suggestions",
)
def refactor(
    target: str,
    config: Optional[str],
    profile: Optional[str],
    environment: Optional[str],
    preview: bool,
    types: tuple,
    feedback: bool,
) -> None:
    """
    Refactor code with intelligent transformations.

    TARGET: Path to file or directory to refactor
    """
    # Setup logging
    _setup_logging()

    console.print()
    _auth_banner("Refactoring")
    console.print()

    # Setup
    target_path = _validate_path(target)
    cfg = _load_config(config, profile, environment)
    _print_refactor_filters(types)
    _confirm_apply_mode(preview)

    # Create backup before applying changes (only in apply mode)
    session_id = None
    if not preview and cfg.backup_enabled:
        try:
            backup_root = target_path.parent if target_path.is_file() else target_path
            backup_system = BackupRollbackSystem(backup_root)

            if target_path.is_file():
                files = [target_path]
            else:
                files = list(target_path.rglob("*.py"))

            if files:
                session_id, failed_files = backup_system.prepare_for_refactoring(
                    files=files,
                    description=f"refactoring {target}",
                    create_git_commit=backup_system.git.is_git_repo(),
                )
                console.print(f"[dim]Backup created: {session_id}[/dim]")
                if failed_files:
                    console.print(
                        f"[yellow]{len(failed_files)} file(s) could not be backed up[/yellow]"
                    )
        except (OSError, PermissionError) as e:
            console.print(f"[yellow]Backup creation failed (I/O error): {e}[/yellow]")
            if not click.confirm("Continue without backup?"):
                raise SystemExit(0)
        except Exception as e:
            console.print(f"[yellow]Backup creation failed: {type(e).__name__}: {e}[/yellow]")
            if not click.confirm("Continue without backup?"):
                raise SystemExit(0)

    # Run refactoring
    try:
        with console.status("[primary]Analyzing and generating refactorings...[/primary]"):
            refactron = Refactron(cfg)
            result = refactron.refactor(
                target,
                preview=preview,
                operation_types=list(types) if types else None,
            )
    except Exception as e:
        console.print(f"[red]Refactoring failed: {e}[/red]")
        raise SystemExit(1)

    # Display results
    summary = result.summary()
    console.print(_create_refactor_table(summary))
    console.print()

    _print_refactor_messages(summary, preview)

    if result.operations:
        # Show ranking info if available
        ranked_count = sum(1 for op in result.operations if "ranking_score" in op.metadata)
        if ranked_count > 0:
            console.print(f"[dim]{ranked_count} operations ranked by learned patterns[/dim]\n")

        console.print("[bold]Refactoring Operations:[/bold]\n")
        console.print(result.show_diff())

        # Record feedback
        if not preview:
            # Auto-record as accepted when applying changes
            _record_applied_operations(refactron, result)
        elif feedback:
            # Interactive feedback collection in preview mode
            _collect_feedback_interactive(refactron, result)

    if session_id and not preview:
        console.print("\n[dim]Tip: Run 'refactron rollback' to undo these changes[/dim]")


@main.command()
@click.argument("operation_id", type=str)
@click.option(
    "--action",
    "-a",
    type=click.Choice(["accepted", "rejected", "ignored"], case_sensitive=False),
    required=True,
    help="Feedback action: accepted, rejected, or ignored",
)
@click.option(
    "--reason",
    "-r",
    type=str,
    help="Optional reason for the feedback",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def feedback(operation_id: str, action: str, reason: Optional[str], config: Optional[str]) -> None:
    """
    Provide feedback on a refactoring operation.

    OPERATION_ID: The unique identifier of the refactoring operation

    Examples:
      refactron feedback abc-123 --action accepted --reason "Improved readability"
      refactron feedback xyz-789 --action rejected --reason "Too risky"
    """
    console.print()
    _auth_banner("Feedback")
    console.print()

    # Load config
    cfg = _load_config(config, None, None)

    # Initialize Refactron
    try:
        refactron = Refactron(cfg)
    except Exception as e:
        console.print(f"[red]Failed to initialize Refactron: {e}[/red]")
        raise SystemExit(1)

    # Record feedback
    try:
        # Check if operation_id exists in recent feedback (for validation)
        if refactron.pattern_storage:
            existing_feedbacks = refactron.pattern_storage.load_feedback()
            operation_exists = any(f.operation_id == operation_id for f in existing_feedbacks)
            if not operation_exists:
                console.print(
                    f"[warning]Warning: Operation ID '{operation_id}' "
                    "not found in recent operations.[/warning]"
                )
                console.print(
                    "[dim]This may be a new or mistyped operation ID. "
                    "Feedback will still be recorded.[/dim]\n"
                )

        refactron.record_feedback(
            operation_id=operation_id,
            action=action.lower(),
            reason=reason,
        )
        console.print(f"[success]Feedback recorded for operation {operation_id}[/success]")
        console.print(f"[dim]Action: {action}[/dim]")
        if reason:
            console.print(f"[dim]Reason: {reason}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to record feedback: {e}[/red]")
        raise SystemExit(1)


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help="Configuration profile to use (dev, staging, prod)",
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help="Environment to use (overrides profile)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "html"]),
    default="text",
    help="Report format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
def report(
    target: str,
    config: Optional[str],
    profile: Optional[str],
    environment: Optional[str],
    format: str,
    output: Optional[str],
) -> None:
    """
    Generate a detailed technical debt report.

    TARGET: Path to file or directory to analyze
    """
    console.print()
    _auth_banner("Report")
    console.print()

    target_path = Path(target)

    # Validate target
    if not target_path.exists():
        console.print(f"[error]Error: Path does not exist: {target}[/error]")
        raise SystemExit(1)

    cfg = _load_config(config, profile, environment)
    cfg.report_format = format

    console.print(f"[secondary]Format: {format.upper()}[/secondary]")

    try:
        with console.status("[primary]Analyzing code and generating report...[/primary]"):
            refactron = Refactron(cfg)
            result = refactron.analyze(target)

        report_content = result.report(detailed=True)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(report_content)

            file_size = output_path.stat().st_size
            console.print(f"\nReport saved to: [bold]{output}[/bold]")
            console.print(f"[dim]Size: {file_size:,} bytes[/dim]")
        else:
            console.print(report_content)

    except Exception as e:
        console.print(f"[red]Report generation failed: {e}[/red]")
        raise SystemExit(1)


@main.command()
@click.argument("target", type=click.Path(exists=True))
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--profile",
    "-p",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help=(
        "Named configuration profile to use (dev, staging, prod). "
        "Profiles typically group config defaults; if both --profile and "
        "--environment are set, the environment determines the final "
        "effective configuration."
    ),
)
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["dev", "staging", "prod"], case_sensitive=False),
    help=(
        "Target runtime environment (dev, staging, prod). When both "
        "--profile and --environment are provided, the environment "
        "overrides the selected profile."
    ),
)
@click.option(
    "--preview/--apply",
    default=True,
    help="Preview fixes or apply them",
)
@click.option(
    "--safety-level",
    "-s",
    type=click.Choice(["safe", "low", "moderate", "high"], case_sensitive=False),
    default="safe",
    help="Maximum risk level for automatic fixes",
)
def autofix(
    target: str,
    config: Optional[str],
    profile: Optional[str],
    environment: Optional[str],
    preview: bool,
    safety_level: str,
) -> None:
    """
    Automatically fix code issues (Phase 3 feature).

    TARGET: Path to file or directory to fix

    Examples:
      refactron autofix myfile.py --preview
      refactron autofix myproject/ --apply --safety-level moderate
    """
    console.print()
    _auth_banner("Auto-fix")
    console.print()

    # Setup
    target_path = _validate_path(target)
    _load_config(config, profile, environment)
    _print_file_count(target_path)

    # Map safety level
    safety_map = {
        "safe": FixRiskLevel.SAFE,
        "low": FixRiskLevel.LOW,
        "moderate": FixRiskLevel.MODERATE,
        "high": FixRiskLevel.HIGH,
    }
    safety = safety_map[safety_level.lower()]

    # Initialize auto-fix engine
    engine = AutoFixEngine(safety_level=safety)

    if preview:
        console.print("[warning]Preview mode: No changes will be applied[/warning]\n")
    else:
        console.print("[success]Apply mode: Changes will be written to files[/success]\n")

    console.print(f"[secondary]Safety level: {safety_level}[/secondary]")
    console.print(f"[secondary]Available fixers: {len(engine.fixers)}[/secondary]\n")

    # Display available fixers
    console.print("[primary bold]Available Auto-fixes:[/primary bold]\n")
    for fixer_name, fixer in engine.fixers.items():
        if fixer.risk_score == 0.0:
            risk_label = "[success]LOW[/success]"
        elif fixer.risk_score < 0.5:
            risk_label = "[warning]MED[/warning]"
        else:
            risk_label = "[error]HIGH[/error]"
        console.print(f"{risk_label} {fixer_name} (risk: {fixer.risk_score:.1f})")

    console.print(
        "\n[secondary]Tip: Auto-fix requires analyzed issues. Integration with analyzers "
        "coming soon![/secondary]"
    )
    console.print(
        "[secondary]For now, use 'refactron analyze' to find issues, then 'refactron refactor' "
        "to fix them.[/secondary]"
    )


@main.command()
@click.option(
    "--template",
    "-t",
    type=click.Choice(["base", "django", "fastapi", "flask"], case_sensitive=False),
    default="base",
    help="Configuration template to use (base, django, fastapi, flask)",
)
def init(template: str) -> None:
    """Initialize Refactron configuration in the current directory."""
    from refactron.core.config_templates import ConfigTemplates

    config_path = Path(".refactron.yaml")

    if config_path.exists():
        console.print("[yellow]Configuration file already exists![/yellow]")
        if not click.confirm("Overwrite?"):
            return

    # Detect project type and suggest appropriate template
    detected_type = _detect_project_type()
    if detected_type and detected_type != template:
        console.print(f"[yellow]Detected {detected_type} project[/yellow]")
        if template == "base":
            console.print(
                f"[yellow]   Consider using --template {detected_type} for "
                f"framework-specific settings[/yellow]"
            )
        else:
            console.print(
                f"[yellow]   Note: Using {template} template, but detected {detected_type}[/yellow]"
            )

    try:
        template_dict = ConfigTemplates.get_template(template)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(template_dict, f, default_flow_style=False, sort_keys=False)

        console.print(f"Created configuration file: {config_path}")
        console.print(f"[dim]Using template: {template}[/dim]")
        if template != "base":
            console.print(
                f"[dim]Template includes framework-specific settings for {template}[/dim]"
            )
        console.print("\n[dim]Edit this file to customize Refactron behavior.[/dim]")
        console.print(
            "[dim]Use --profile or --environment options to switch between dev/staging/prod[/dim]"
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)


@main.command()
@click.option(
    "--session",
    "-s",
    type=str,
    default=None,
    help="Specific session ID to rollback (default: latest session)",
)
@click.option(
    "--use-git",
    is_flag=True,
    default=False,
    help="Use Git rollback instead of file backup",
)
@click.option(
    "--list",
    "list_sessions",
    is_flag=True,
    default=False,
    help="List all backup sessions",
)
@click.option(
    "--clear",
    is_flag=True,
    default=False,
    help="Clear all backup sessions",
)
def rollback(
    session: Optional[str],
    use_git: bool,
    list_sessions: bool,
    clear: bool,
) -> None:
    """
    Rollback refactoring changes to restore original files.

    By default, restores files from the latest backup session.
    Use --session to specify a specific session ID.
    Use --use-git to rollback using Git instead of file backups.

    Examples:
      refactron rollback              # Rollback latest session
      refactron rollback --list       # List all backup sessions
      refactron rollback --session session_20240101_120000
      refactron rollback --use-git    # Use Git rollback
      refactron rollback --clear      # Clear all backups
    """
    console.print("\n🔄 [bold blue]Refactron Rollback[/bold blue]\n")

    system = BackupRollbackSystem()

    if list_sessions:
        sessions = system.list_sessions()
        if not sessions:
            console.print("[yellow]No backup sessions found.[/yellow]")
            return

        table = Table(title="Backup Sessions", show_header=True, header_style="bold magenta")
        table.add_column("Session ID", style="cyan")
        table.add_column("Timestamp", style="green")
        table.add_column("Files", justify="right")
        table.add_column("Git Commit", style="dim")
        table.add_column("Description")

        for sess in sessions:
            git_commit = sess.get("git_commit", "")
            if git_commit:
                git_commit = git_commit[:8] + "..."
            table.add_row(
                sess["id"],
                sess["timestamp"],
                str(len(sess["files"])),
                git_commit or "N/A",
                sess.get("description", "")[:30],
            )

        console.print(table)
        return

    if clear:
        if not click.confirm("Are you sure you want to clear all backup sessions?"):
            return

        count = system.clear_all()
        console.print(f"[success]Cleared {count} backup session(s).[/success]")
        return

    sessions = system.list_sessions()
    if not sessions:
        console.print("[yellow]No backup sessions found.[/yellow]")
        console.print("[dim]Tip: Backups are created automatically when using --apply mode.[/dim]")
        return

    if session:
        sess = system.backup_manager.get_session(session)
        if not sess:
            console.print(f"[error]Session not found: {session}[/error]")
            console.print("[dim]Use 'refactron rollback --list' to see available sessions.[/dim]")
            raise SystemExit(1)
        console.print(f"[dim]Rolling back session: {session}[/dim]")
        console.print(f"[dim]Files to restore: {len(sess['files'])}[/dim]")
    else:
        latest = sessions[-1]
        console.print(f"[dim]Rolling back latest session: {latest['id']}[/dim]")
        console.print(f"[dim]Files to restore: {len(latest['files'])}[/dim]")

    if use_git:
        console.print("[dim]Using Git rollback...[/dim]")
    else:
        console.print("[dim]Using file backup rollback...[/dim]")

    console.print(
        "\n[warning]This will overwrite your current files with backup versions.[/warning]"
    )
    if not click.confirm("Are you sure you want to proceed with rollback?"):
        console.print("[yellow]Rollback cancelled.[/yellow]")
        return

    result = system.rollback(session_id=session, use_git=use_git)

    if result["success"]:
        console.print(f"\n[success]{result['message']}[/success]")
        if result.get("files_restored"):
            console.print(f"[dim]Files restored: {result['files_restored']}[/dim]")
        if result.get("failed_files"):
            console.print(
                f"[warning]Failed to restore: {', '.join(result['failed_files'])}[/warning]"
            )
    else:
        console.print(f"\n[red]Rollback failed: {result['message']}[/red]")
        raise SystemExit(1)


@main.command()
@click.option(
    "--enable",
    "action",
    flag_value="enable",
    help="Enable telemetry collection",
)
@click.option(
    "--disable",
    "action",
    flag_value="disable",
    help="Disable telemetry collection",
)
@click.option(
    "--status",
    "action",
    flag_value="status",
    default=True,
    help="Show telemetry status (default)",
)
def telemetry(action: str) -> None:
    """
    Manage opt-in telemetry settings.

    Telemetry helps us understand how Refactron is used in real-world scenarios
    and improve its performance. All data is anonymous and no code or personal
    information is collected.

    Examples:
      refactron telemetry --status   # Show current status
      refactron telemetry --enable   # Enable telemetry
      refactron telemetry --disable  # Disable telemetry
    """
    from refactron.core.telemetry import TelemetryConfig

    console.print()
    _auth_banner("Telemetry")
    console.print()

    config = TelemetryConfig()

    if action == "enable":
        config.enable()
        console.print("[success]Telemetry has been enabled.[/success]")
        console.print("\n[dim]Thank you for helping improve Refactron![/dim]")
        console.print("[dim]Only anonymous usage statistics are collected.[/dim]")
        console.print(f"[dim]Anonymous ID: {config.anonymous_id}[/dim]")
    elif action == "disable":
        config.disable()
        console.print("[warning]Telemetry has been disabled.[/warning]")
        console.print(
            "\n[dim]You can re-enable it anytime with 'refactron telemetry --enable'[/dim]"
        )
    else:  # status
        if config.enabled:
            console.print("[success]Telemetry is currently enabled[/success]")
            console.print(f"\n[dim]Anonymous ID: {config.anonymous_id}[/dim]")
            console.print("\n[bold]What data is collected:[/bold]")
            console.print("  • Number of files analyzed")
            console.print("  • Analysis execution time")
            console.print("  • Number of issues found (not the actual issues)")
            console.print("  • Python version and OS platform")
            console.print("  • Refactoring operations applied")
            console.print("\n[bold]What is NOT collected:[/bold]")
            console.print("  • Your code or file names")
            console.print("  • Personal information")
            console.print("  • Specific error messages or stack traces")
            console.print("\n[dim]Use --disable to turn off telemetry[/dim]")
        else:
            console.print("[yellow]Telemetry is currently disabled[/yellow]")
            console.print("\n[dim]Use --enable to help improve Refactron[/dim]")


@main.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format",
)
def metrics(format: str) -> None:
    """
    Display collected metrics from the current session.

    Shows performance metrics, analyzer hit counts, and other statistics
    from Refactron operations.

    Examples:
      refactron metrics              # Show metrics in text format
      refactron metrics --format json  # Show metrics in JSON format
    """
    import json as json_module

    from refactron.core.metrics import get_metrics_collector

    console.print()
    _auth_banner("Metrics")
    console.print()

    collector = get_metrics_collector()
    summary = collector.get_combined_summary()

    if format == "json":
        console.print(json_module.dumps(summary, indent=2))
    else:
        # Text format
        analysis = summary.get("analysis", {})
        refactoring = summary.get("refactoring", {})

        # Analysis metrics
        console.print("[bold]Analysis Metrics:[/bold]")
        console.print(f"  Files analyzed: {analysis.get('total_files_analyzed', 0)}")
        console.print(f"  Files failed: {analysis.get('total_files_failed', 0)}")
        console.print(f"  Issues found: {analysis.get('total_issues_found', 0)}")
        console.print(f"  Total time: {analysis.get('total_analysis_time_ms', 0):.2f}ms")
        console.print(f"  Avg time per file: {analysis.get('average_time_per_file_ms', 0):.2f}ms")
        console.print(f"  Success rate: {analysis.get('success_rate_percent', 0):.1f}%")

        # Analyzer hit counts
        analyzer_hits = analysis.get("analyzer_hit_counts", {})
        if analyzer_hits:
            console.print("\n[bold]Analyzer Hit Counts:[/bold]")
            for analyzer, count in sorted(analyzer_hits.items()):
                console.print(f"  {analyzer}: {count}")

        # Refactoring metrics
        console.print("\n[bold]Refactoring Metrics:[/bold]")
        console.print(f"  Applied: {refactoring.get('total_refactorings_applied', 0)}")
        console.print(f"  Failed: {refactoring.get('total_refactorings_failed', 0)}")
        console.print(f"  Total time: {refactoring.get('total_refactoring_time_ms', 0):.2f}ms")
        console.print(f"  Success rate: {refactoring.get('success_rate_percent', 0):.1f}%")

        # Refactorer hit counts
        refactorer_hits = refactoring.get("refactorer_hit_counts", {})
        if refactorer_hits:
            console.print("\n[bold]Refactorer Hit Counts:[/bold]")
            for refactorer, count in sorted(refactorer_hits.items()):
                console.print(f"  {refactorer}: {count}")


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind Prometheus metrics server to (default: 127.0.0.1 for localhost-only)",
)
@click.option(
    "--port",
    default=9090,
    type=int,
    help="Port for Prometheus metrics server",
)
def serve_metrics(host: str, port: int) -> None:
    """
    Start a Prometheus metrics HTTP server.

    This command starts a persistent HTTP server that exposes Refactron metrics
    in Prometheus format on the /metrics endpoint.

    Examples:
      refactron serve-metrics                    # Start on 0.0.0.0:9090
      refactron serve-metrics --port 8080        # Start on port 8080
      refactron serve-metrics --host 127.0.0.1   # Bind to localhost only
    """
    from refactron.core.prometheus_metrics import start_metrics_server

    console.print()
    _auth_banner("Metrics Server")
    console.print()

    try:
        start_metrics_server(host=host, port=port)
        console.print(f"[success]Metrics server started on http://{host}:{port}[/success]")
        console.print("\n[dim]Endpoints:[/dim]")
        console.print(f"[dim]  • http://{host}:{port}/metrics - Prometheus metrics[/dim]")
        console.print(f"[dim]  • http://{host}:{port}/health  - Health check[/dim]")
        console.print("\n[warning]Press Ctrl+C to stop the server[/warning]")

        # Keep the server running
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n\n[warning]Stopping metrics server...[/warning]")
            from refactron.core.prometheus_metrics import stop_metrics_server

            stop_metrics_server()
            console.print("[success]Metrics server stopped[/success]")
    except Exception as e:
        console.print(f"[red]Failed to start metrics server: {e}[/red]")
        raise SystemExit(1)


@main.command()
@click.argument("type", type=click.Choice(["github", "gitlab", "pre-commit", "all"]))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory (default: current directory)",
)
@click.option(
    "--python-versions",
    default="3.8,3.9,3.10,3.11,3.12",
    help="Comma-separated Python versions (default: 3.8,3.9,3.10,3.11,3.12)",
)
@click.option(
    "--fail-on-critical/--no-fail-on-critical",
    default=True,
    help="Fail build on critical issues (default: True)",
)
@click.option(
    "--fail-on-errors/--no-fail-on-errors",
    default=False,
    help="Fail build on error-level issues (default: False)",
)
@click.option(
    "--max-critical",
    default=0,
    type=int,
    help="Maximum allowed critical issues (default: 0)",
)
@click.option(
    "--max-errors",
    default=10,
    type=int,
    help="Maximum allowed error-level issues (default: 10)",
)
def generate_cicd(
    type: str,
    output: Optional[str],
    python_versions: str,
    fail_on_critical: bool,
    fail_on_errors: bool,
    max_critical: int,
    max_errors: int,
) -> None:
    """
    Generate CI/CD integration templates.

    TYPE: Type of template to generate (github, gitlab, pre-commit, all)

    Examples:
      refactron generate-cicd github --output .github/workflows
      refactron generate-cicd gitlab --output .
      refactron generate-cicd pre-commit --output .
      refactron generate-cicd all --output .
    """
    from pathlib import Path

    from refactron.cicd.github_actions import GitHubActionsGenerator
    from refactron.cicd.gitlab_ci import GitLabCIGenerator
    from refactron.cicd.pre_commit import PreCommitGenerator

    console.print()
    _auth_banner("CI/CD Templates")
    console.print()

    output_path = Path(output) if output else Path(".")

    # Parse Python versions
    python_vers = [v.strip() for v in python_versions.split(",")]

    # Quality gate configuration
    quality_gate = {
        "fail_on_critical": fail_on_critical,
        "fail_on_errors": fail_on_errors,
        "max_critical": max_critical,
        "max_errors": max_errors,
    }

    try:
        if type in ("github", "all"):
            console.print("[dim]Generating GitHub Actions workflow...[/dim]")
            github_gen = GitHubActionsGenerator()

            # Create workflows directory
            workflows_dir = output_path / ".github" / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)

            # Generate main analysis workflow
            workflow_content = github_gen.generate_analysis_workflow(
                python_versions=python_vers, quality_gate=quality_gate
            )
            workflow_path = workflows_dir / "refactron-analysis.yml"
            github_gen.save_workflow(workflow_content, workflow_path)
            console.print(f"[success]Created: {workflow_path}[/success]")

            # Generate pre-commit workflow
            pre_commit_workflow = github_gen.generate_pre_commit_workflow(
                python_version=python_vers[0] if python_vers else "3.11"
            )
            pre_commit_path = workflows_dir / "refactron-pre-commit.yml"
            github_gen.save_workflow(pre_commit_workflow, pre_commit_path)
            console.print(f"[success]Created: {pre_commit_path}[/success]")

        if type in ("gitlab", "all"):
            console.print("[dim]Generating GitLab CI pipeline...[/dim]")
            gitlab_gen = GitLabCIGenerator()

            # Generate main pipeline
            pipeline_content = gitlab_gen.generate_analysis_pipeline(
                python_versions=python_vers, quality_gate=quality_gate
            )
            pipeline_path = output_path / ".gitlab-ci.yml"
            gitlab_gen.save_pipeline(pipeline_content, pipeline_path)
            console.print(f"[success]Created: {pipeline_path}[/success]")

            # Generate pre-commit pipeline
            pre_commit_pipeline = gitlab_gen.generate_pre_commit_pipeline(
                python_version=python_vers[0] if python_vers else "3.11"
            )
            pre_commit_pipeline_path = output_path / ".gitlab-ci-pre-commit.yml"
            gitlab_gen.save_pipeline(pre_commit_pipeline, pre_commit_pipeline_path)
            console.print(f"[success]Created: {pre_commit_pipeline_path}[/success]")

        if type in ("pre-commit", "all"):
            console.print("[dim]Generating pre-commit configuration...[/dim]")
            pre_commit_gen = PreCommitGenerator()

            # Generate pre-commit config
            config_content = pre_commit_gen.generate_pre_commit_config(
                fail_on_critical=fail_on_critical,
                fail_on_errors=fail_on_errors,
                max_critical=max_critical,
                max_errors=max_errors,
            )
            config_path = output_path / ".pre-commit-config.refactron.yaml"
            pre_commit_gen.save_config(config_content, config_path)
            console.print(f"[success]Created: {config_path}[/success]")

            # Generate simple hook script (only if this is a git repository)
            git_dir = output_path / ".git"
            if git_dir.is_dir():
                hook_content = pre_commit_gen.generate_simple_hook()
                hooks_dir = git_dir / "hooks"
                hooks_dir.mkdir(parents=True, exist_ok=True)
                hook_path = hooks_dir / "pre-commit.refactron"
                pre_commit_gen.save_hook(hook_content, hook_path)
                console.print(f"[success]Created: {hook_path}[/success]")
            else:
                console.print(
                    "[dim]No .git directory found at the output path; "
                    "skipping installation of the git hook script.[/dim]"
                )

        console.print("\n[success]CI/CD templates generated successfully![/success]")
        console.print("\n[dim]Next steps:[/dim]")
        console.print("[dim]  1. Review and customize the generated templates[/dim]")
        console.print("[dim]  2. For GitHub Actions: Workflows are in .github/workflows/[/dim]")
        console.print("[dim]  3. For GitLab CI: Merge into your .gitlab-ci.yml[/dim]")
        console.print("[dim]  4. For pre-commit: Install with 'pre-commit install'[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to generate templates: {e}[/red]")
        raise SystemExit(1)


@main.group()
def patterns() -> None:
    """Pattern learning and project-specific tuning commands."""


def _get_pattern_storage_from_config(config: RefactronConfig) -> PatternStorage:
    """
    Initialize PatternStorage for CLI commands.

    This helper centralizes how we construct PatternStorage so that tuning
    commands behave consistently with the rest of the system.
    """
    try:
        return PatternStorage()
    except Exception as e:
        console.print(f"[red]Failed to initialize pattern storage: {e}[/red]")
        raise SystemExit(1)


@patterns.command("analyze")
@click.option(
    "--project",
    "-p",
    "project_path",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Path to project root to analyze (default: current directory)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def patterns_analyze(project_path: str, config_path: Optional[str]) -> None:
    """
    Analyze learned patterns for a specific project.

    Shows project-specific acceptance rates and usage statistics.
    """
    console.print()
    _auth_banner("Pattern Analysis")
    console.print()

    cfg = _load_config(config_path, None, None)
    _setup_logging()

    project_root = Path(project_path).resolve()

    storage = _get_pattern_storage_from_config(cfg)
    tuner = RuleTuner(storage=storage)

    try:
        analysis = tuner.analyze_project_patterns(project_root)
    except Exception as e:
        console.print(f"[red]Failed to analyze project patterns: {e}[/red]")
        raise SystemExit(1)

    patterns = analysis.get("patterns", [])
    if not patterns:
        console.print("[yellow]No pattern feedback found for this project yet.[/yellow]")
        return

    table = Table(
        title=f"Patterns for project: {analysis['project_path']}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Pattern ID", style="cyan", no_wrap=True)
    table.add_column("Op Type", style="green")
    table.add_column("Proj Acc%", justify="right")
    table.add_column("Proj Decisions", justify="right")
    table.add_column("Global Acc%", justify="right")
    table.add_column("Enabled", justify="center")
    table.add_column("Weight", justify="right")

    for p in patterns:
        proj_acc = p["project_acceptance_rate"] * 100.0
        glob_acc = p["global_acceptance_rate"] * 100.0
        table.add_row(
            p["pattern_id"],
            p["operation_type"],
            f"{proj_acc:.1f}",
            str(p["project_total_decisions"]),
            f"{glob_acc:.1f}",
            "Yes" if p["enabled"] else "No",
            f"{p['weight']:.2f}",
        )

    console.print(table)


@patterns.command("recommend")
@click.option(
    "--project",
    "-p",
    "project_path",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Path to project root to analyze (default: current directory)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def patterns_recommend(project_path: str, config_path: Optional[str]) -> None:
    """
    Show rule tuning recommendations for a project.

    Recommendations are based on project-specific pattern acceptance rates.
    """
    console.print()
    _auth_banner("Tuning Recommendations")
    console.print()

    cfg = _load_config(config_path, None, None)
    _setup_logging()

    project_root = Path(project_path).resolve()

    storage = _get_pattern_storage_from_config(cfg)
    tuner = RuleTuner(storage=storage)

    try:
        recs = tuner.generate_recommendations(project_root)
    except Exception as e:
        console.print(f"[red]Failed to generate recommendations: {e}[/red]")
        raise SystemExit(1)

    to_disable = recs.get("to_disable", [])
    to_enable = recs.get("to_enable", [])
    weights = recs.get("weights", {})

    if not to_disable and not to_enable and not weights:
        console.print("[yellow]No tuning recommendations available yet for this project.[/yellow]")
        return

    table = Table(
        title=f"Tuning Recommendations for: {recs['project_path']}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Pattern ID", style="cyan", no_wrap=True)
    table.add_column("Action", style="green")
    table.add_column("Weight", justify="right")

    for pattern_id in sorted(set(to_disable) | set(to_enable) | set(weights.keys())):
        if pattern_id in to_disable:
            action = "disable"
        elif pattern_id in to_enable:
            action = "enable"
        else:
            action = "adjust_weight"

        weight_str = f"{weights[pattern_id]:.2f}" if pattern_id in weights else "-"
        table.add_row(pattern_id, action, weight_str)

    console.print(table)


@patterns.command("tune")
@click.option(
    "--project",
    "-p",
    "project_path",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Path to project root to tune (default: current directory)",
)
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Automatically apply all recommended tuning without confirmation",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def patterns_tune(
    project_path: str,
    auto: bool,
    config_path: Optional[str],
) -> None:
    """
    Apply tuning recommendations to the project profile.

    By default, shows recommended changes and asks for confirmation.
    Use --auto to apply without prompting.
    """
    console.print()
    _auth_banner("Apply Tuning")
    console.print()

    cfg = _load_config(config_path, None, None)
    _setup_logging()

    project_root = Path(project_path).resolve()

    storage = _get_pattern_storage_from_config(cfg)
    tuner = RuleTuner(storage=storage)

    try:
        recs = tuner.generate_recommendations(project_root)
    except Exception as e:
        console.print(f"[red]Failed to generate recommendations: {e}[/red]")
        raise SystemExit(1)

    to_disable = recs.get("to_disable", [])
    to_enable = recs.get("to_enable", [])
    weights = recs.get("weights", {})

    if not to_disable and not to_enable and not weights:
        console.print("[yellow]No tuning recommendations to apply for this project.[/yellow]")
        return

    console.print("[bold]Planned changes:[/bold]")
    if to_disable:
        console.print(f"  • Disable patterns: [warning]{', '.join(sorted(to_disable))}[/warning]")
    if to_enable:
        console.print(f"  • Enable patterns: [success]{', '.join(sorted(to_enable))}[/success]")
    if weights:
        console.print("  • Adjust weights:")
        for pid, w in sorted(weights.items()):
            console.print(f"    - {pid}: {w:.2f}")

    if not auto:
        if not click.confirm("\nApply these tuning changes?"):
            console.print("[dim]No changes applied.[/dim]")
            return

    try:
        profile = tuner.apply_tuning(project_root, recs)
    except Exception as e:
        console.print(f"[red]Failed to apply tuning: {e}[/red]")
        raise SystemExit(1)

    console.print(
        f"\n[success]Applied tuning for project [bold]{profile.project_path}[/bold][/success] "
        f"(profile ID: {profile.project_id})"
    )


@patterns.command("profile")
@click.option(
    "--project",
    "-p",
    "project_path",
    type=click.Path(exists=True, file_okay=False),
    default=".",
    help="Path to project root (default: current directory)",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def patterns_profile(project_path: str, config_path: Optional[str]) -> None:
    """
    Show the current pattern profile for a project.

    Includes enabled/disabled patterns and custom weights.
    """
    console.print()
    _auth_banner("Project Profile")
    console.print()

    cfg = _load_config(config_path, None, None)
    _setup_logging()

    project_root = Path(project_path).resolve()

    storage = _get_pattern_storage_from_config(cfg)

    try:
        profile = storage.get_project_profile(project_root)
    except Exception as e:
        console.print(f"[red]Failed to load project profile: {e}[/red]")
        raise SystemExit(1)

    console.print(f"Project ID: [bold]{profile.project_id}[/bold]")
    console.print(f"Project Path: [bold]{profile.project_path}[/bold]")
    console.print(f"Last Updated: [dim]{profile.last_updated.isoformat()}[/dim]\n")

    table = Table(
        title="Pattern Profile",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Pattern ID", style="cyan", no_wrap=True)
    table.add_column("Enabled", justify="center")
    table.add_column("Weight", justify="right")

    all_pattern_ids = (
        set(profile.enabled_patterns)
        | set(profile.disabled_patterns)
        | set(profile.pattern_weights.keys())
    )

    if not all_pattern_ids:
        console.print("[yellow]No project-specific tuning has been applied yet.[/yellow]")
        return

    for pattern_id in sorted(all_pattern_ids):
        enabled = profile.is_pattern_enabled(pattern_id)
        weight = profile.get_pattern_weight(pattern_id, default=1.0)
        table.add_row(
            pattern_id,
            "Yes" if enabled else "No",
            f"{weight:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    main(prog_name="refactron")
