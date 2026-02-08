"""Tests for the Groq LLM client."""

import os
import pytest
from unittest.mock import Mock, patch

from refactron.llm.client import GroqClient


class TestGroqClient:
    """Test cases for GroqClient."""

    @pytest.fixture
    def mock_groq_api(self):
        """Mock the Groq API."""
        with patch("refactron.llm.client.Groq") as mock_groq:
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Generated text response"

            # Mock client
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            yield mock_groq

    def test_client_initialization_with_api_key(self, mock_groq_api):
        """Test client initialization with explicit API key."""
        client = GroqClient(api_key="test_key_123")

        assert client.api_key == "test_key_123"
        assert client.model == "llama-3.3-70b-versatile"
        assert client.temperature == 0.2
        assert client.max_tokens == 2000

    def test_client_initialization_from_env(self, mock_groq_api):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "env_key_456"}):
            client = GroqClient()
            assert client.api_key == "env_key_456"

    def test_client_initialization_no_api_key(self, mock_groq_api):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                GroqClient()

    def test_generate_basic(self, mock_groq_api):
        """Test basic text generation."""
        client = GroqClient(api_key="testkey")
        response = client.generate("Hello, how are you?")

        assert response == "Generated text response"
        assert client.client.chat.completions.create.called

    def test_generate_with_system_prompt(self, mock_groq_api):
        """Test generation with system prompt."""
        client = GroqClient(api_key="testkey")
        response = client.generate(prompt="User message", system="You are a helpful assistant")

        assert response == "Generated text response"

        # Check that both system and user messages were sent
        call_args = client.client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_generate_with_custom_params(self, mock_groq_api):
        """Test generation with custom temperature and max_tokens."""
        client = GroqClient(api_key="testkey")
        client.generate(prompt="Test", temperature=0.5, max_tokens=1000)

        call_args = client.client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 1000

    def test_custom_model(self, mock_groq_api):
        """Test initialization with custom model."""
        client = GroqClient(api_key="testkey", model="llama-3.1-8b-instant")

        assert client.model == "llama-3.1-8b-instant"

    def test_check_health_success(self, mock_groq_api):
        """Test health check when API is accessible."""
        client = GroqClient(api_key="testkey")
        is_healthy = client.check_health()

        assert is_healthy is True

    def test_check_health_failure(self, mock_groq_api):
        """Test health check when API fails."""
        client = GroqClient(api_key="testkey")
        client.client.chat.completions.create.side_effect = Exception("API Error")

        is_healthy = client.check_health()
        assert is_healthy is False
