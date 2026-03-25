"""Coverage-focused tests for low-coverage core modules."""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from refactron.core import device_auth, repositories, workspace
from refactron.core.credentials import RefactronCredentials
from refactron.core.device_auth import DeviceAuthorization
from refactron.core.workspace import WorkspaceManager, WorkspaceMapping


class _FakeHttpResponse:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _valid_creds(expires_at=None) -> RefactronCredentials:
    return RefactronCredentials(
        api_base_url="https://api.refactron.dev",
        access_token="token",
        token_type="Bearer",
        expires_at=expires_at,
        email="a@b.com",
        plan="pro",
    )


def test_workspace_add_get_list_remove(tmp_path: Path) -> None:
    mgr = WorkspaceManager(config_path=tmp_path / "workspaces.json")
    mapping = WorkspaceMapping(
        repo_name="demo",
        repo_full_name="user/demo",
        local_path=str(tmp_path / "demo"),
        connected_at="2026-03-24T00:00:00Z",
        repo_id=42,
    )

    mgr.add_workspace(mapping)
    assert mgr.get_workspace("user/demo").repo_name == "demo"
    assert mgr.get_workspace("demo").repo_full_name == "user/demo"
    assert mgr.get_workspace_by_path(str(tmp_path / "demo")).repo_full_name == "user/demo"
    assert len(mgr.list_workspaces()) == 1
    assert mgr.remove_workspace("user/demo") is True
    assert mgr.remove_workspace("user/demo") is False


def test_workspace_load_invalid_json_returns_empty(tmp_path: Path) -> None:
    cfg = tmp_path / "workspaces.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{broken", encoding="utf-8")
    mgr = WorkspaceManager(config_path=cfg)
    assert mgr.list_workspaces() == []


def test_workspace_detect_repository_https_ssh_and_missing(tmp_path: Path) -> None:
    mgr = WorkspaceManager(config_path=tmp_path / "workspaces.json")

    https_repo = tmp_path / "repo_https"
    (https_repo / ".git").mkdir(parents=True)
    (https_repo / ".git" / "config").write_text(
        '[remote "origin"]\nurl = https://github.com/acme/project.git\n',
        encoding="utf-8",
    )
    assert mgr.detect_repository(https_repo) == "acme/project"

    ssh_repo = tmp_path / "repo_ssh"
    (ssh_repo / ".git").mkdir(parents=True)
    (ssh_repo / ".git" / "config").write_text(
        '[remote "origin"]\nurl = git@github.com:acme/project2.git\n',
        encoding="utf-8",
    )
    assert mgr.detect_repository(ssh_repo) == "acme/project2"

    missing = tmp_path / "no_repo"
    missing.mkdir()
    assert mgr.detect_repository(missing) is None


def test_list_repositories_success_for_list_and_dict(monkeypatch) -> None:
    monkeypatch.setattr(repositories, "load_credentials", lambda: _valid_creds())

    payloads = [
        json.dumps(
            [
                {
                    "id": 1,
                    "name": "repo",
                    "full_name": "u/repo",
                    "description": None,
                    "private": False,
                    "html_url": "https://github.com/u/repo",
                    "clone_url": "https://github.com/u/repo.git",
                    "ssh_url": "git@github.com:u/repo.git",
                    "default_branch": "main",
                    "language": "Python",
                    "updated_at": "2026-03-24T00:00:00Z",
                }
            ]
        ),
        json.dumps({"repositories": []}),
    ]

    def fake_urlopen(req, timeout=10):  # noqa: ARG001
        return _FakeHttpResponse(payloads.pop(0))

    monkeypatch.setattr(repositories, "urlopen", fake_urlopen)
    repos = repositories.list_repositories("https://api.refactron.dev/")
    assert len(repos) == 1
    assert repos[0].full_name == "u/repo"
    repos2 = repositories.list_repositories("https://api.refactron.dev/")
    assert repos2 == []


def test_list_repositories_auth_and_error_paths(monkeypatch) -> None:
    monkeypatch.setattr(repositories, "load_credentials", lambda: None)
    with pytest.raises(RuntimeError, match="Not authenticated"):
        repositories.list_repositories("https://api.refactron.dev")

    monkeypatch.setattr(
        repositories,
        "load_credentials",
        lambda: _valid_creds(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1)),
    )
    with pytest.raises(RuntimeError, match="session has expired"):
        repositories.list_repositories("https://api.refactron.dev")

    monkeypatch.setattr(repositories, "load_credentials", lambda: _valid_creds())

    err = HTTPError(
        url="https://api.refactron.dev/api/github/repositories",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(b'{"message":"bad token"}'),
    )

    def raise_401(req, timeout=10):  # noqa: ARG001
        raise err

    monkeypatch.setattr(repositories, "urlopen", raise_401)
    with pytest.raises(RuntimeError, match="Authentication failed"):
        repositories.list_repositories("https://api.refactron.dev")

    def raise_network(req, timeout=10):  # noqa: ARG001
        raise URLError("down")

    monkeypatch.setattr(repositories, "urlopen", raise_network)
    with pytest.raises(RuntimeError, match="Network error"):
        repositories.list_repositories("https://api.refactron.dev")


def test_device_auth_helpers_and_start_authorization(monkeypatch) -> None:
    assert (
        device_auth._normalize_base_url("https://api.refactron.dev/") == "https://api.refactron.dev"
    )

    monkeypatch.setattr(
        device_auth,
        "_post_json",
        lambda *a, **k: {
            "device_code": "dc",
            "user_code": "uc",
            "verification_uri": "https://example.com",
            "expires_in": "900",
            "interval": "0",
        },
    )
    auth = device_auth.start_device_authorization()
    assert isinstance(auth, DeviceAuthorization)
    assert auth.interval == 1

    monkeypatch.setattr(device_auth, "_post_json", lambda *a, **k: {"device_code": "x"})
    with pytest.raises(RuntimeError, match="Invalid /oauth/device response"):
        device_auth.start_device_authorization()


def test_poll_for_token_pending_slowdown_success_and_expired(monkeypatch) -> None:
    calls = []
    responses = [
        {"error": "authorization_pending"},
        {"error": "slow_down"},
        {
            "access_token": "tok",
            "token_type": "Bearer",
            "expires_in": 1200,
            "user": {"email": "x@y.com"},
        },
    ]

    def fake_post(*a, **k):  # noqa: ARG001
        return responses.pop(0)

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(device_auth, "_post_json", fake_post)
    token = device_auth.poll_for_token(
        device_code="dc",
        interval_seconds=1,
        expires_in_seconds=30,
        sleep_fn=fake_sleep,
    )
    assert token.access_token == "tok"
    assert calls == [1, 6]

    monkeypatch.setattr(device_auth, "_post_json", lambda *a, **k: {"error": "expired_token"})
    with pytest.raises(RuntimeError, match="Device code expired"):
        device_auth.poll_for_token("dc", expires_in_seconds=5, sleep_fn=lambda *_: None)
