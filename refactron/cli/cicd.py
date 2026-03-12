"""
Refactron CLI - CI/CD Module.
Commands for CI/CD integration, initialization, and feedback.
"""

from pathlib import Path
from typing import Optional

import click

from refactron import Refactron
from refactron.cicd.github_actions import GitHubActionsGenerator
from refactron.cicd.gitlab_ci import GitLabCIGenerator
from refactron.cicd.pre_commit import PreCommitGenerator
from refactron.cli.ui import _auth_banner, console
from refactron.cli.utils import _detect_project_type, _load_config
from refactron.core.config_templates import ConfigTemplates


@click.command()
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


@click.command()
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
def feedback(
    operation_id: str,
    action: str,
    reason: Optional[str],
    config: Optional[str],
) -> None:
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


@click.command()
@click.option(
    "--template",
    "-t",
    type=click.Choice(["base", "django", "fastapi", "flask"], case_sensitive=False),
    default="base",
    help="Configuration template to use (base, django, fastapi, flask)",
)
def init(template: str) -> None:
    """Initialize Refactron configuration in the current directory."""
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

        # Import yaml locally to avoid top-level dependency if not needed elsewhere
        import yaml

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
