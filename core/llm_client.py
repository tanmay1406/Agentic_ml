"""
core/llm_client.py

===============================================================================
                                SynapseAI
===============================================================================

Unified interface for interacting with local LLMs.

Currently supports:
    ✓ Ollama

Future:
    ✓ OpenAI
    ✓ Anthropic
    ✓ Gemini

All agents should communicate ONLY through this class.

===============================================================================
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests


class LLMConnectionError(Exception):
    """Raised when the LLM server cannot be reached."""


class LLMGenerationError(Exception):
    """Raised when generation fails."""


class LLMClient:
    """
    Wrapper around the Ollama API.

    Example
    -------
    >>> llm = LLMClient()
    >>> response = llm.generate("Hello")
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        timeout: int = 120,
        retries: int = 3,
    ):

        self.model = model
        self.host = host.rstrip("/")

        self.temperature = temperature

        self.timeout = timeout

        self.retries = retries

        self.session = requests.Session()

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    @staticmethod
    def _extract_code(text: str) -> str:
        """
        Extract Python code from markdown if present.
        """

        pattern = r"```(?:python)?\n?(.*?)```"

        match = re.search(
            pattern,
            text,
            re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return text.strip()

    def _request(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        Internal request to Ollama.
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        last_error = None

        for attempt in range(self.retries):

            try:

                response = self.session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                return response.json()["response"]

            except requests.exceptions.RequestException as e:

                last_error = e

                time.sleep(1)

        raise LLMConnectionError(
            f"Unable to connect to Ollama.\n{last_error}"
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        General text generation.
        """

        start = time.perf_counter()

        response = self._request(
            prompt,
            system_prompt,
        )

        latency = time.perf_counter() - start

        print(
            f"[LLM] Generated response in "
            f"{latency:.2f}s"
        )

        return response.strip()

    def generate_code(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        Generate Python code.

        Automatically strips markdown formatting.
        """

        response = self.generate(
            prompt,
            system_prompt,
        )

        return self._extract_code(response)

    # ==========================================================
    # Utilities
    # ==========================================================

    def health_check(self) -> bool:
        """
        Check whether Ollama server is alive.
        """

        try:

            response = self.session.get(
                f"{self.host}/api/tags",
                timeout=5,
            )

            return response.status_code == 200

        except Exception:

            return False

    def list_models(self):
        """
        Return installed Ollama models.
        """

        response = self.session.get(
            f"{self.host}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return [
            model["name"]
            for model in data.get("models", [])
        ]

    def change_model(
        self,
        model: str,
    ):
        """
        Switch model without recreating the client.
        """

        self.model = model

    def __repr__(self):

        return (
            f"LLMClient("
            f"model='{self.model}', "
            f"host='{self.host}')"
        )