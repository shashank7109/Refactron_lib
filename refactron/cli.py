"""Command-line interface for Refactron."""

import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table

from refactron import Refactron
from refactron.autofix.engine import AutoFixEngine
from refactron.autofix.models import FixRiskLevel
from refactron.core.analysis_result import AnalysisResult
from refactron.core.backup import BackupRollbackSystem
from refactron.core.config import RefactronConfig
from refactron.core.exceptions import ConfigError

console = Console()


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
            console.print(f"[dim]📄 Loading config from: {config_path}[/dim]")
            if profile or environment:
                env_display = environment or profile
                console.print(f"[dim]🎯 Using profile/environment: {env_display}[/dim]")
            return RefactronConfig.from_file(Path(config_path), profile, environment)
        return RefactronConfig.default()
    except ConfigError as e:
        console.print(f"[red]❌ Configuration Error: {e}[/red]")
        if e.recovery_suggestion:
            console.print(f"[yellow]💡 {e.recovery_suggestion}[/yellow]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]❌ Unexpected error loading configuration: {e}[/red]")
        raise SystemExit(1)


def _validate_path(target: str) -> Path:
    """Validate target path exists."""
    target_path = Path(target)
    if not target_path.exists():
        console.print(f"[red]❌ Error: Path does not exist: {target}[/red]")
        raise SystemExit(1)
    return target_path


def _print_file_count(target_path: Path) -> None:
    """Print count of Python files if target is directory."""
    if target_path.is_dir():
        py_files = list(target_path.rglob("*.py"))
        console.print(f"[dim]📁 Found {len(py_files)} Python file(s) to analyze[/dim]\n")


def _create_summary_table(summary: dict) -> Table:
    """Create analysis summary table."""
    table = Table(title="Analysis Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Files Found", str(summary["total_files"]))
    table.add_row("Files Analyzed", str(summary["files_analyzed"]))
    if summary.get("files_failed", 0) > 0:
        table.add_row("Files Failed", str(summary["files_failed"]))
    table.add_row("Total Issues", str(summary["total_issues"]))
    table.add_row("🔴 Critical", str(summary["critical"]))
    table.add_row("❌ Errors", str(summary["errors"]))
    table.add_row("⚡ Warnings", str(summary["warnings"]))
    table.add_row("ℹ️  Info", str(summary["info"]))

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
            f"[yellow]⚠️  {summary['files_failed']} file(s) failed analysis "
            f"and were skipped[/yellow]"
        )

    if summary["total_issues"] == 0 and summary.get("files_failed", 0) == 0:
        console.print("[green]✨ Excellent! No issues found.[/green]")
    elif summary["total_issues"] == 0 and summary.get("files_failed", 0) > 0:
        console.print(
            "[yellow]⚠️  No issues found in analyzed files, but some files failed.[/yellow]"
        )
    elif summary["critical"] > 0:
        console.print(
            f"[red]⚠️  Found {summary['critical']} critical issue(s) that need immediate "
            f"attention![/red]"
        )


def _print_detailed_issues(result: AnalysisResult) -> None:
    """Print detailed issues list."""
    console.print("[bold]Detailed Issues:[/bold]\n")
    level_icons = {
        "critical": "🔴",
        "error": "❌",
        "warning": "⚡",
        "info": "ℹ️",
    }

    for issue in result.all_issues:
        icon = level_icons.get(issue.level.value, "•")
        console.print(f"{icon} {issue}")
        if issue.suggestion:
            console.print(f"   [dim]💡 {issue.suggestion}[/dim]")
        console.print()


def _print_helpful_tips(summary: dict, detailed: bool) -> None:
    """Print helpful tips based on results."""
    if summary["total_issues"] > 0 and not detailed:
        console.print("[dim]💡 Tip: Use --detailed to see all issues[/dim]")

    if summary["total_issues"] > 5:
        console.print(
            "[dim]💡 Tip: Run 'refactron refactor --preview' to see suggested fixes[/dim]"
        )


def _print_refactor_filters(types: tuple) -> None:
    """Print operation type filters if specified."""
    if types:
        console.print(f"[dim]🎯 Filtering for: {', '.join(types)}[/dim]\n")


def _confirm_apply_mode(preview: bool) -> None:
    """Warn and confirm if using --apply mode."""
    if not preview:
        console.print("[yellow]⚠️  --apply mode will modify your files![/yellow]")
        if not click.confirm("Continue?"):
            raise SystemExit(0)


def _create_refactor_table(summary: dict) -> Table:
    """Create refactoring summary table."""
    table = Table(title="Refactoring Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Operations", str(summary["total_operations"]))
    table.add_row("Safe Operations", str(summary["safe"]))
    table.add_row("High Risk", str(summary["high_risk"]))
    table.add_row("Applied", "✅ Yes" if summary["applied"] else "❌ No")

    return table


def _print_refactor_messages(summary: dict, preview: bool) -> None:
    """Print status messages for refactoring results."""
    if summary["total_operations"] == 0:
        console.print("[green]✨ No refactoring opportunities found. Your code looks good![/green]")
    elif summary["high_risk"] > 0:
        console.print(
            f"[yellow]⚠️  {summary['high_risk']} operation(s) are high-risk. Review "
            f"carefully![/yellow]"
        )

    if preview and summary["total_operations"] > 0:
        console.print("\n[yellow]ℹ️  This is a preview. Use --apply to apply changes.[/yellow]")
        console.print("[dim]💡 Tip: Review each change carefully before applying[/dim]")

    if summary["total_operations"] > 0 and summary["applied"]:
        console.print("\n[green]✅ Refactoring completed! Don't forget to test your code.[/green]")


@click.group()
@click.version_option(version="1.0.1")
def main() -> None:
    """
    Refactron - The Intelligent Code Refactoring Transformer

    Analyze, refactor, and optimize your Python code with ease.
    """
    pass


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

    console.print("\n🔍 [bold blue]Refactron Analysis[/bold blue]\n")

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
        with console.status("[bold green]🔎 Analyzing code...[/bold green]"):
            refactron = Refactron(cfg)
            result = refactron.analyze(target)
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
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

        console.print("\n[bold]📊 Metrics Summary:[/bold]")
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
def refactor(
    target: str,
    config: Optional[str],
    profile: Optional[str],
    environment: Optional[str],
    preview: bool,
    types: tuple,
) -> None:
    """
    Refactor code with intelligent transformations.

    TARGET: Path to file or directory to refactor
    """
    # Setup logging
    _setup_logging()

    console.print("\n🔧 [bold blue]Refactron Refactoring[/bold blue]\n")

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
                console.print(f"[dim]📦 Backup created: {session_id}[/dim]")
                if failed_files:
                    console.print(
                        f"[yellow]⚠️  {len(failed_files)} file(s) could not be backed up[/yellow]"
                    )
        except (OSError, PermissionError) as e:
            console.print(f"[yellow]⚠️  Backup creation failed (I/O error): {e}[/yellow]")
            if not click.confirm("Continue without backup?"):
                raise SystemExit(0)
        except Exception as e:
            console.print(f"[yellow]⚠️  Backup creation failed: {type(e).__name__}: {e}[/yellow]")
            if not click.confirm("Continue without backup?"):
                raise SystemExit(0)

    # Run refactoring
    try:
        with console.status("[bold green]🔎 Analyzing and generating refactorings...[/bold green]"):
            refactron = Refactron(cfg)
            result = refactron.refactor(
                target,
                preview=preview,
                operation_types=list(types) if types else None,
            )
    except Exception as e:
        console.print(f"[red]❌ Refactoring failed: {e}[/red]")
        raise SystemExit(1)

    # Display results
    summary = result.summary()
    console.print(_create_refactor_table(summary))
    console.print()

    _print_refactor_messages(summary, preview)

    if result.operations:
        console.print("[bold]Refactoring Operations:[/bold]\n")
        console.print(result.show_diff())

    if session_id and not preview:
        console.print("\n[dim]💡 Tip: Run 'refactron rollback' to undo these changes[/dim]")


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
    console.print("\n📊 [bold blue]Generating Report[/bold blue]\n")

    target_path = Path(target)

    # Validate target
    if not target_path.exists():
        console.print(f"[red]❌ Error: Path does not exist: {target}[/red]")
        raise SystemExit(1)

    cfg = _load_config(config, profile, environment)
    cfg.report_format = format

    console.print(f"[dim]📝 Format: {format.upper()}[/dim]")

    try:
        with console.status("[bold green]📊 Analyzing code and generating report...[/bold green]"):
            refactron = Refactron(cfg)
            result = refactron.analyze(target)

        report_content = result.report(detailed=True)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(report_content)

            file_size = output_path.stat().st_size
            console.print(f"\n✅ Report saved to: [bold]{output}[/bold]")
            console.print(f"[dim]📦 Size: {file_size:,} bytes[/dim]")
        else:
            console.print(report_content)

    except Exception as e:
        console.print(f"[red]❌ Report generation failed: {e}[/red]")
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
    console.print("\n🔧 [bold blue]Refactron Auto-fix[/bold blue]\n")

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
        console.print("[yellow]📋 Preview mode: No changes will be applied[/yellow]\n")
    else:
        console.print("[green]✅ Apply mode: Changes will be written to files[/green]\n")

    console.print(f"[dim]🛡️  Safety level: {safety_level}[/dim]")
    console.print(f"[dim]🔧 Available fixers: {len(engine.fixers)}[/dim]\n")

    # Display available fixers
    console.print("[bold]Available Auto-fixes:[/bold]\n")
    for fixer_name, fixer in engine.fixers.items():
        risk_emoji = "🟢" if fixer.risk_score == 0.0 else "🟡" if fixer.risk_score < 0.5 else "🔴"
        console.print(f"{risk_emoji} {fixer_name} (risk: {fixer.risk_score:.1f})")

    console.print(
        "\n[dim]💡 Tip: Auto-fix requires analyzed issues. Integration with analyzers "
        "coming soon![/dim]"
    )
    console.print(
        "[dim]📖 For now, use 'refactron analyze' to find issues, then 'refactron refactor' "
        "to fix them.[/dim]"
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
        console.print("[yellow]⚠️  Configuration file already exists![/yellow]")
        if not click.confirm("Overwrite?"):
            return

    # Detect project type and suggest appropriate template
    detected_type = _detect_project_type()
    if detected_type and detected_type != template:
        console.print(f"[yellow]💡 Detected {detected_type} project[/yellow]")
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

        console.print(f"✅ Created configuration file: {config_path}")
        console.print(f"[dim]📋 Using template: {template}[/dim]")
        if template != "base":
            console.print(
                f"[dim]💡 Template includes framework-specific settings for {template}[/dim]"
            )
        console.print("\n[dim]Edit this file to customize Refactron behavior.[/dim]")
        console.print(
            "[dim]💡 Use --profile or --environment options to switch between dev/staging/prod[/dim]"
        )
    except ValueError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
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
        console.print(f"[green]✅ Cleared {count} backup session(s).[/green]")
        return

    sessions = system.list_sessions()
    if not sessions:
        console.print("[yellow]No backup sessions found.[/yellow]")
        console.print(
            "[dim]💡 Tip: Backups are created automatically when using --apply mode.[/dim]"
        )
        return

    if session:
        sess = system.backup_manager.get_session(session)
        if not sess:
            console.print(f"[red]❌ Session not found: {session}[/red]")
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
        "\n[yellow]⚠️  This will overwrite your current files with backup versions.[/yellow]"
    )
    if not click.confirm("Are you sure you want to proceed with rollback?"):
        console.print("[yellow]Rollback cancelled.[/yellow]")
        return

    result = system.rollback(session_id=session, use_git=use_git)

    if result["success"]:
        console.print(f"\n[green]✅ {result['message']}[/green]")
        if result.get("files_restored"):
            console.print(f"[dim]Files restored: {result['files_restored']}[/dim]")
        if result.get("failed_files"):
            console.print(
                f"[yellow]⚠️  Failed to restore: {', '.join(result['failed_files'])}[/yellow]"
            )
    else:
        console.print(f"\n[red]❌ Rollback failed: {result['message']}[/red]")
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

    console.print("\n📊 [bold blue]Refactron Telemetry[/bold blue]\n")

    config = TelemetryConfig()

    if action == "enable":
        config.enable()
        console.print("[green]✅ Telemetry has been enabled.[/green]")
        console.print("\n[dim]Thank you for helping improve Refactron![/dim]")
        console.print("[dim]Only anonymous usage statistics are collected.[/dim]")
        console.print(f"[dim]Anonymous ID: {config.anonymous_id}[/dim]")
    elif action == "disable":
        config.disable()
        console.print("[yellow]Telemetry has been disabled.[/yellow]")
        console.print(
            "\n[dim]You can re-enable it anytime with 'refactron telemetry --enable'[/dim]"
        )
    else:  # status
        if config.enabled:
            console.print("[green]✅ Telemetry is currently enabled[/green]")
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

    console.print("\n📈 [bold blue]Refactron Metrics[/bold blue]\n")

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

    console.print("\n🚀 [bold blue]Starting Prometheus Metrics Server[/bold blue]\n")

    try:
        start_metrics_server(host=host, port=port)
        console.print(f"[green]✅ Metrics server started on http://{host}:{port}[/green]")
        console.print("\n[dim]Endpoints:[/dim]")
        console.print(f"[dim]  • http://{host}:{port}/metrics - Prometheus metrics[/dim]")
        console.print(f"[dim]  • http://{host}:{port}/health  - Health check[/dim]")
        console.print("\n[yellow]Press Ctrl+C to stop the server[/yellow]")

        # Keep the server running
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Stopping metrics server...[/yellow]")
            from refactron.core.prometheus_metrics import stop_metrics_server

            stop_metrics_server()
            console.print("[green]✅ Metrics server stopped[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to start metrics server: {e}[/red]")
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

    console.print("\n🔧 [bold blue]Generating CI/CD Templates[/bold blue]\n")

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
            console.print("[dim]📝 Generating GitHub Actions workflow...[/dim]")
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
            console.print(f"[green]✅ Created: {workflow_path}[/green]")

            # Generate pre-commit workflow
            pre_commit_workflow = github_gen.generate_pre_commit_workflow(
                python_version=python_vers[0] if python_vers else "3.11"
            )
            pre_commit_path = workflows_dir / "refactron-pre-commit.yml"
            github_gen.save_workflow(pre_commit_workflow, pre_commit_path)
            console.print(f"[green]✅ Created: {pre_commit_path}[/green]")

        if type in ("gitlab", "all"):
            console.print("[dim]📝 Generating GitLab CI pipeline...[/dim]")
            gitlab_gen = GitLabCIGenerator()

            # Generate main pipeline
            pipeline_content = gitlab_gen.generate_analysis_pipeline(
                python_versions=python_vers, quality_gate=quality_gate
            )
            pipeline_path = output_path / ".gitlab-ci.yml"
            gitlab_gen.save_pipeline(pipeline_content, pipeline_path)
            console.print(f"[green]✅ Created: {pipeline_path}[/green]")

            # Generate pre-commit pipeline
            pre_commit_pipeline = gitlab_gen.generate_pre_commit_pipeline(
                python_version=python_vers[0] if python_vers else "3.11"
            )
            pre_commit_pipeline_path = output_path / ".gitlab-ci-pre-commit.yml"
            gitlab_gen.save_pipeline(pre_commit_pipeline, pre_commit_pipeline_path)
            console.print(f"[green]✅ Created: {pre_commit_pipeline_path}[/green]")

        if type in ("pre-commit", "all"):
            console.print("[dim]📝 Generating pre-commit configuration...[/dim]")
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
            console.print(f"[green]✅ Created: {config_path}[/green]")

            # Generate simple hook script (only if this is a git repository)
            git_dir = output_path / ".git"
            if git_dir.is_dir():
                hook_content = pre_commit_gen.generate_simple_hook()
                hooks_dir = git_dir / "hooks"
                hooks_dir.mkdir(parents=True, exist_ok=True)
                hook_path = hooks_dir / "pre-commit.refactron"
                pre_commit_gen.save_hook(hook_content, hook_path)
                console.print(f"[green]✅ Created: {hook_path}[/green]")
            else:
                console.print(
                    "[dim]ℹ No .git directory found at the output path; "
                    "skipping installation of the git hook script.[/dim]"
                )

        console.print("\n[green]✅ CI/CD templates generated successfully![/green]")
        console.print("\n[dim]💡 Next steps:[/dim]")
        console.print("[dim]  1. Review and customize the generated templates[/dim]")
        console.print("[dim]  2. For GitHub Actions: Workflows are in .github/workflows/[/dim]")
        console.print("[dim]  3. For GitLab CI: Merge into your .gitlab-ci.yml[/dim]")
        console.print("[dim]  4. For pre-commit: Install with 'pre-commit install'[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Failed to generate templates: {e}[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
