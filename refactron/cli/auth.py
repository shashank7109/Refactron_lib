"""
Refactron CLI - Authentication Module.
Commands for logging in, logging out, checking status, and managing telemetry.
"""

import time
import webbrowser
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import click
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from refactron.cli.ui import _auth_banner, console
from refactron.cli.utils import _setup_logging, _validate_api_key
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
from refactron.core.telemetry import disable_telemetry, enable_telemetry, get_telemetry_collector


@click.command()
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
                Panel(
                    f"Failed to start login: {e}",
                    title="Connection Error",
                    border_style="red",
                )
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
        with console.status(
            "[bold blue]Waiting for authorization...[/bold blue]",
            spinner="dots",
        ):
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
                    "API key is required for this plan.",
                    title="Login aborted",
                    border_style="red",
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
            Panel(
                f"Failed to save credentials: {e}",
                title="Error",
                border_style="error",
            )
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
            summary,
            title="Login Successful",
            border_style="success",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


@click.command()
def logout() -> None:
    """Log out of Refactron CLI."""
    _setup_logging()
    path = credentials_path()

    if delete_credentials(path):
        console.print(
            Panel(
                f"Credentials removed from {path}",
                title="Logged Out",
                border_style="success",
                box=box.ROUNDED,
            )
        )
    else:
        console.print(f"[yellow]No credentials found at {path}[/yellow]")


@click.group()
def auth() -> None:
    """Manage authentication state."""
    pass


@auth.command(name="status")
def auth_status() -> None:
    """Show current authentication status."""
    _setup_logging()
    creds = load_credentials()

    if not creds:
        console.print("\n[yellow]Not logged in.[/yellow]")
        console.print("Run 'refactron login' to authenticate.")
        return

    _auth_banner("Auth Status")

    status_color = "green"
    status_text = "Active"

    if creds.expires_at:
        now = datetime.now(timezone.utc)
        if creds.expires_at < now:
            status_color = "red"
            status_text = "Expired"

    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Key", style="secondary")
    table.add_column("Value", style="bold white")

    table.add_row("Status", f"[{status_color}]{status_text}[/{status_color}]")
    table.add_row("User", creds.email or "Unknown")
    table.add_row("Plan", (creds.plan or "Unknown").upper())
    table.add_row("API URL", creds.api_base_url)

    if creds.expires_at:
        expires = creds.expires_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        table.add_row("Expires", expires)

    console.print(table)


# Alias logout under auth group
auth.add_command(logout, name="logout")


@click.command()
@click.option("--enable", is_flag=True, help="Enable telemetry collection")
@click.option("--disable", is_flag=True, help="Disable telemetry collection")
@click.option(
    "--status",
    is_flag=True,
    default=True,
    help="Show current telemetry status",
)
def telemetry(enable: bool, disable: bool, status: bool) -> None:
    """Manage telemetry settings."""
    if enable:
        enable_telemetry()
        console.print(
            "[success]Telemetry enabled. Thank you for helping improve Refactron![/success]"
        )
    elif disable:
        disable_telemetry()
        console.print("[warning]Telemetry disabled.[/warning]")
    else:
        # Status
        collector = get_telemetry_collector(enabled=True)  # Get instance to check config
        is_enabled = collector.enabled
        status_color = "green" if is_enabled else "yellow"
        status_text = "Enabled" if is_enabled else "Disabled"

        console.print(f"\nTelemetry is [{status_color}]{status_text}[/{status_color}]")
        if is_enabled:
            console.print("[dim]Anonymous usage data is being collected.[/dim]")
        else:
            console.print("[dim]No usage data is being collected.[/dim]")
