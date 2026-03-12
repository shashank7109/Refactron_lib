"""
Refactron CLI - Patterns Module.
Commands for pattern learning and project-specific tuning.
"""

from pathlib import Path
from typing import Optional

import click
from rich.table import Table

from refactron.cli.ui import _auth_banner, console
from refactron.cli.utils import _get_pattern_storage_from_config, _load_config, _setup_logging
from refactron.patterns.tuner import RuleTuner


@click.group()
def patterns() -> None:
    """Pattern learning and project-specific tuning commands."""
    pass


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
