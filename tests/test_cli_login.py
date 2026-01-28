"""Tests for CLI login (device-code flow)."""

from pathlib import Path

from click.testing import CliRunner

from refactron.cli import ApiKeyValidationResult, main
from refactron.core.credentials import RefactronCredentials
from refactron.core.device_auth import DeviceAuthorization, TokenResponse


def test_login_device_code_flow(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    # Pretend there are no existing credentials so login flow runs
    monkeypatch.setattr("refactron.cli.load_credentials", lambda: None)

    # Stub device authorization response
    def _mock_start_device_authorization(api_base_url: str, timeout_seconds: int = 10):
        assert api_base_url == "http://0.0.0.0:3001"
        return DeviceAuthorization(
            device_code="devcode-123",
            user_code="ABCD-EFGH",
            verification_uri="https://refactron.dev/auth/device",
            expires_in=900,
            interval=1,
        )

    # Stub polling response
    def _mock_poll_for_token(
        device_code: str,
        api_base_url: str,
        interval_seconds: int,
        expires_in_seconds: int,
        timeout_seconds: int = 10,
    ):
        assert device_code == "devcode-123"
        assert api_base_url == "http://0.0.0.0:3001"
        return TokenResponse(
            access_token="jwt.token.here",
            token_type="Bearer",
            expires_in=3600,
            email="user@example.com",
            plan="pro",
            api_key="ref_ABC123",
        )

    saved: dict = {}
    prompts: dict = {}

    def _mock_save_credentials(creds: RefactronCredentials) -> None:
        saved["creds"] = creds

    def _mock_credentials_path() -> Path:
        return tmp_path / "credentials.json"

    def _mock_prompt(text: str, hide_input: bool = False, default: str = "") -> str:
        prompts["text"] = text
        prompts["hide_input"] = hide_input
        return "ref_TESTKEY"

    monkeypatch.setattr(
        "refactron.cli.start_device_authorization", _mock_start_device_authorization
    )
    monkeypatch.setattr("refactron.cli.poll_for_token", _mock_poll_for_token)
    monkeypatch.setattr("refactron.cli.save_credentials", _mock_save_credentials)
    monkeypatch.setattr("refactron.cli.credentials_path", _mock_credentials_path)
    monkeypatch.setattr("refactron.cli.click.prompt", _mock_prompt)
    # API key should be verified before being stored
    monkeypatch.setattr(
        "refactron.cli._validate_api_key",
        lambda *args, **kwargs: ApiKeyValidationResult(ok=True, message="Verified."),
    )

    result = runner.invoke(main, ["login", "--no-browser", "--api-base-url", "http://0.0.0.0:3001"])
    assert result.exit_code == 0, result.output
    assert "Refactron" in result.output
    assert "Login" in result.output
    assert "ABCD-EFGH" in result.output
    assert "app.refactron.dev/login" in result.output
    assert "Login complete" in result.output
    assert "user@example.com" in result.output
    assert "pro" in result.output

    assert "creds" in saved
    creds: RefactronCredentials = saved["creds"]
    assert creds.api_base_url == "http://0.0.0.0:3001"
    assert creds.access_token == "jwt.token.here"
    # API key should come from user prompt, not from the backend response
    assert creds.api_key == "ref_TESTKEY"


def test_logout_no_credentials(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr("refactron.cli.delete_credentials", lambda: False)

    result = runner.invoke(main, ["logout"])
    assert result.exit_code == 0
    assert "No stored credentials found" in result.output


def test_auth_status_not_logged_in(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr("refactron.cli.load_credentials", lambda: None)

    result = runner.invoke(main, ["auth", "status"])
    assert result.exit_code == 0
    assert "Not logged in." in result.output


def test_login_skips_when_already_logged_in(monkeypatch, tmp_path: Path) -> None:
    """Login should not start device flow when valid creds exist unless --force is used."""
    runner = CliRunner()

    fake_creds = RefactronCredentials(
        api_base_url="http://0.0.0.0:3001",
        access_token="existing-token",
        token_type="Bearer",
        expires_at=None,
        email="existing@example.com",
        plan="pro",
        api_key="ref_EXISTING",
    )

    # load_credentials returns an existing valid credential
    monkeypatch.setattr("refactron.cli.load_credentials", lambda: fake_creds)

    # Ensure device flow is NOT called
    monkeypatch.setattr(
        "refactron.cli.start_device_authorization",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    result = runner.invoke(main, ["login", "--no-browser"])
    assert result.exit_code == 0
    assert "Already authenticated" in result.output
    assert "existing@example.com" in result.output


def test_login_does_not_save_invalid_api_key(monkeypatch, tmp_path: Path) -> None:
    """If API key fails verification, login should abort and not store it."""
    runner = CliRunner()

    monkeypatch.setattr("refactron.cli.load_credentials", lambda: None)

    def _mock_start_device_authorization(api_base_url: str, timeout_seconds: int = 10):
        return DeviceAuthorization(
            device_code="devcode-123",
            user_code="ABCD-EFGH",
            verification_uri="https://refactron.dev/auth/device",
            expires_in=900,
            interval=1,
        )

    def _mock_poll_for_token(
        device_code: str,
        api_base_url: str,
        interval_seconds: int,
        expires_in_seconds: int,
        timeout_seconds: int = 10,
    ):
        return TokenResponse(
            access_token="jwt.token.here",
            token_type="Bearer",
            expires_in=3600,
            email="user@example.com",
            plan="pro",
            api_key=None,
        )

    saved: dict = {}

    def _mock_save_credentials(creds: RefactronCredentials) -> None:
        saved["creds"] = creds

    def _mock_credentials_path() -> Path:
        return tmp_path / "credentials.json"

    def _mock_prompt(text: str, hide_input: bool = False, default: str = "") -> str:
        return "ref_INVALID"

    monkeypatch.setattr(
        "refactron.cli.start_device_authorization", _mock_start_device_authorization
    )
    monkeypatch.setattr("refactron.cli.poll_for_token", _mock_poll_for_token)
    monkeypatch.setattr("refactron.cli.save_credentials", _mock_save_credentials)
    monkeypatch.setattr("refactron.cli.credentials_path", _mock_credentials_path)
    monkeypatch.setattr("refactron.cli.click.prompt", _mock_prompt)
    # Simulate backend rejecting the key
    monkeypatch.setattr(
        "refactron.cli._validate_api_key",
        lambda *args, **kwargs: ApiKeyValidationResult(ok=False, message="Invalid API key."),
    )

    result = runner.invoke(main, ["login", "--no-browser"])
    assert result.exit_code != 0
    assert "Invalid API key." in result.output
    # Login should have been aborted before saving any credentials
    assert "creds" not in saved
