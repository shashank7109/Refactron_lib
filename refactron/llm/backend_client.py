"""Client for Refactron backend LLM proxy."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from refactron.core.credentials import load_credentials


class BackendLLMClient:
    """Client that proxies LLM requests through Refactron backend."""

    def __init__(
        self,
        backend_url: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ):
        """Initialize backend client.

        Args:
            backend_url: Refactron backend URL
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        # Default to the local testing backend URL if not provided
        self.backend_url = (backend_url or "https://api.refactron.dev").rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load credentials to get access token/API key
        self.creds = load_credentials()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using backend API.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Use API key if available, otherwise use access token
        if self.creds:
            if self.creds.api_key:
                headers["X-API-Key"] = self.creds.api_key
            elif self.creds.access_token:
                headers["Authorization"] = f"Bearer {self.creds.access_token}"

        payload = {
            "prompt": prompt,
            "system": system,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "model": self.model,
        }

        try:
            response = requests.post(
                f"{self.backend_url}/api/llm/generate",
                json=payload,
                headers=headers,
                timeout=60,
            )

            if response.status_code != 200:
                error_msg = response.text
                if response.headers.get("Content-Type") == "application/json":
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", error_msg)
                    except Exception:
                        pass
                raise RuntimeError(f"Backend LLM proxy error ({response.status_code}): {error_msg}")

            data = response.json()
            return data["content"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Refactron backend: {e}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during backend LLM generation: {e}")

    def check_health(self) -> bool:
        """Check if the backend API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            response = requests.get(
                f"{self.backend_url}/api/llm/health",
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False
