"""Tests for BackendLLMClient."""

from unittest.mock import MagicMock, patch

import pytest

from refactron.llm.backend_client import BackendLLMClient


@pytest.fixture
def mock_credentials():
    with patch("refactron.llm.backend_client.load_credentials") as mock:
        creds = MagicMock()
        creds.api_key = "test-api-key"
        creds.access_token = "test-access-token"
        mock.return_value = creds
        yield mock


def test_backend_client_initialization():
    client = BackendLLMClient(backend_url="http://test-backend:3000")
    assert client.backend_url == "http://test-backend:3000"
    assert client.model == "llama-3.3-70b-versatile"


@patch("requests.post")
def test_backend_client_generate(mock_post, mock_credentials):
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": "Suggested code"}
    mock_post.return_value = mock_response

    client = BackendLLMClient(backend_url="http://test-backend:3000")
    result = client.generate(prompt="Refactor this", system="You are an expert")

    assert result == "Suggested code"

    # Verify request
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://test-backend:3000/api/llm/generate"
    assert kwargs["json"]["prompt"] == "Refactor this"
    assert kwargs["json"]["system"] == "You are an expert"
    assert kwargs["headers"]["X-API-Key"] == "test-api-key"


@patch("requests.post")
def test_backend_client_error_handling(mock_post, mock_credentials):
    # Mock error response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    client = BackendLLMClient()
    with pytest.raises(
        RuntimeError, match=r"Backend LLM proxy error \(500\): Internal Server Error"
    ):
        client.generate(prompt="Refactor this")


@patch("requests.get")
def test_backend_client_check_health(mock_get):
    # Mock healthy response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    client = BackendLLMClient()
    assert client.check_health() is True

    # Mock unhealthy response
    mock_response.status_code = 503
    assert client.check_health() is False
