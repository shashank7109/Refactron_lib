"""GitHub repository integration for Refactron CLI.

This module provides functionality to interact with the Refactron backend API
to fetch GitHub repositories connected to the user's account.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from refactron.core.credentials import load_credentials


@dataclass(frozen=True)
class Repository:
    """Represents a GitHub repository."""

    id: int
    name: str
    full_name: str
    description: Optional[str]
    private: bool
    html_url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    language: Optional[str]
    updated_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Repository":
        """Create a Repository instance from API response data."""
        return cls(
            id=data["id"],
            name=data["name"],
            full_name=data["full_name"],
            description=data.get("description"),
            private=data["private"],
            html_url=data["html_url"],
            clone_url=data["clone_url"],
            ssh_url=data["ssh_url"],
            default_branch=data.get("default_branch", "main"),
            language=data.get("language"),
            updated_at=data["updated_at"],
        )


def list_repositories(api_base_url: str, timeout_seconds: int = 10) -> List[Repository]:
    """Fetch all GitHub repositories connected to the user's account.

    Args:
        api_base_url: The Refactron API base URL
        timeout_seconds: Request timeout in seconds

    Returns:
        List of Repository objects

    Raises:
        RuntimeError: If the request fails or user is not authenticated
    """
    # Load credentials
    creds = load_credentials()
    if not creds:
        raise RuntimeError("Not authenticated. Please run 'refactron login' first.")

    if not creds.access_token:
        raise RuntimeError("No access token found. Please run 'refactron login' first.")

    # Check if token is expired
    from datetime import datetime, timezone

    if creds.expires_at:
        try:
            # Parse the expiration time
            from datetime import datetime

            if isinstance(creds.expires_at, str):
                # Remove timezone info for comparison
                expires_str = creds.expires_at.replace("+00:00", "").replace("Z", "")
                expires_at = datetime.fromisoformat(expires_str).replace(tzinfo=timezone.utc)
            else:
                expires_at = creds.expires_at

            now = datetime.now(timezone.utc)
            if now >= expires_at:
                raise RuntimeError("Your session has expired. Please run 'refactron login' again.")
        except (ValueError, AttributeError):
            # If we can't parse the expiration, continue anyway
            pass

    # Normalize the base URL
    base = api_base_url.rstrip("/")
    url = f"{base}/api/github/repositories"

    # Prepare the request with Bearer token
    req = Request(
        url=url,
        headers={
            "Authorization": f"Bearer {creds.access_token}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else []

            # Handle both list and dict wrapper formats
            repositories_data = []
            if isinstance(data, list):
                repositories_data = data
            elif isinstance(data, dict):
                # Try common wrapper keys
                repositories_data = (
                    data.get("repositories") or data.get("data") or data.get("repos") or []
                )
                if not isinstance(repositories_data, list):
                    raise RuntimeError(
                        f"Unexpected API response format. Expected list or dict with 'repositories' key. "
                        f"Got: {type(data)} with keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
                    )
            else:
                raise RuntimeError(f"Unexpected API response type: {type(data)}")

            return [Repository.from_dict(repo) for repo in repositories_data]

    except HTTPError as e:
        if e.code == 401:
            # Try to get more details from the error response
            try:
                error_body = e.read().decode("utf-8")
                error_data = json.loads(error_body)
                detail = error_data.get("message", error_data.get("detail", "Unknown error"))
            except Exception:
                detail = "No additional details"

            raise RuntimeError(
                f"Authentication failed (HTTP 401): {detail}\n\n"
                "Possible causes:\n"
                "  1. Your session has expired - run 'refactron login' again\n"
                "  2. The access token is invalid\n"
                "  3. The API endpoint requires different authentication\n\n"
                f"API URL: {url}\n"
                f"Token present: {'Yes' if creds.access_token else 'No'}"
            )
        elif e.code == 403:
            raise RuntimeError(
                "GitHub access denied. Please reconnect your GitHub account on the Refactron website."
            )
        elif e.code == 404:
            raise RuntimeError(
                "Repository endpoint not found. Please check your API base URL or update Refactron."
            )
        else:
            # Try to parse error message from response
            try:
                error_body = e.read().decode("utf-8")
                error_data = json.loads(error_body)
                message = error_data.get("message", str(e))
            except Exception:
                message = str(e)
            raise RuntimeError(f"Failed to fetch repositories (HTTP {e.code}): {message}")

    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}. Is the Refactron API accessible?")

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from API: {e}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching repositories: {e}")
