"""
Shared UI components for the Refactron CLI.
Contains the Rich theme, console instance, and common display functions.
"""

import platform
import random
import sys
import time
from typing import Any, Optional

import click
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from refactron import Refactron, __version__
from refactron.core.analysis_result import AnalysisResult
from refactron.core.refactor_result import RefactorResult

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
    grid.add_row(
        Text(
            "The Intelligent Code Refactoring Transformer",
            style="secondary italic",
        )
    )

    console.print(
        Panel(
            grid,
            style="panel.border",
            box=box.ROUNDED,
            padding=(1, 2),
            subtitle=f"[secondary]v{__version__}[/secondary]",
            subtitle_align="right",
        )
    )


def _print_file_count(target_path: Any) -> None:
    """Print count of Python files if target is directory."""
    # Note: target_path typed as Any to avoid circular imports or Path issues
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
        "Errors",
        str(summary["errors"]),
        style="bold red" if summary["errors"] > 0 else "dim",
    )
    table.add_row(
        "Warnings",
        str(summary["warnings"]),
        style="bold yellow" if summary["warnings"] > 0 else "dim",
    )
    table.add_row(
        "Info",
        str(summary["info"]),
        style="cyan" if summary["info"] > 0 else "dim",
    )

    return table


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

            info_table.add_row("Version:", f"v{__version__}")
            info_table.add_row("Python:", sys.version.split()[0])
            info_table.add_row("OS:", platform.system())

            grid.add_row(Text("\n"))
            grid.add_row(Align.center(info_table))

            # Tip
            grid.add_row(Text("\n"))
            grid.add_row(
                Align.center(
                    Panel(
                        Text(selected_tip, style="cyan"),
                        border_style="#333333",
                        expand=False,
                    )
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
    console.print(Align.center(Text(f"v{__version__}", style="dim")))
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
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
        expand=True,
        padding=(0, 2),
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


def _run_minimal_loop(ctx: click.Context) -> None:
    """Run a minimal interactive loop for help and version only."""
    from refactron.core.credentials import load_credentials

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
                "\n[bold]>[/bold] ",
                choices=["1", "2", "3"],
                default="3",
                show_choices=False,
            )

            if choice == "1":
                _print_custom_help(ctx)
            elif choice == "2":
                console.print(f"\nRefactron CLI v{__version__}")
            elif choice == "3":
                console.print("Goodbye!")
                break

        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break


def _interactive_file_selector(directory: Any, pattern: str = "*.py") -> Optional[Any]:
    """
    Interactively select a file from a directory.

    Args:
        directory: Path-like object
        pattern: Glob pattern to filter files

    Returns:
        Selected file Path or None if aborted/skipped
    """
    # Import locally to avoid circular dependencies
    from pathlib import Path

    target_dir = Path(directory)
    if not target_dir.exists():
        return None

    files = list(target_dir.rglob(pattern))
    if not files:
        console.print("[warning]No matching files found.[/warning]")
        return None

    # Limit to top 20 to avoid overwhelming UI
    display_files = files[:20]

    table = Table(title=f"Files in {target_dir.name}", box=box.SIMPLE)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Path", style="white")

    for i, path in enumerate(display_files, 1):
        rel_path = path.relative_to(target_dir)
        table.add_row(str(i), str(rel_path))

    console.print(table)
    if len(files) > 20:
        console.print(f"[dim]...and {len(files) - 20} more files[/dim]")

    choice = IntPrompt.ask("Select a file (0 to cancel)", default=0, show_default=True)

    if choice == 0 or choice > len(display_files):
        return None

    return display_files[choice - 1]
