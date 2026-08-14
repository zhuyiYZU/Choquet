import json
import os
import ssl
import time
import traceback
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import certifi


DEBUG_LLM = False


class LLMClientError(RuntimeError):
    """Raised when an LLM request fails with a clear, user-facing cause."""


class LLMClient:
    """OpenAI-compatible chat completions client using only HTTP.

    This client targets OpenAI-compatible chat/completions gateways and does not
    import or call provider-specific SDKs.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env: str,
        base_url: Optional[str],
        timeout: int,
        max_retries: int,
        temperature: float,
    ):
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = (base_url or self._default_base_url(provider)).rstrip("/")
        self.timeout = int(timeout)
        self.max_retries = max(0, int(max_retries))
        self.temperature = temperature

    @classmethod
    def from_config(cls) -> "LLMClient":
        from config import (
            LLM_API_KEY_ENV,
            LLM_BASE_URL,
            LLM_MAX_RETRIES,
            LLM_MODEL,
            LLM_PROVIDER,
            LLM_TEMPERATURE,
            LLM_TIMEOUT,
        )

        return cls(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            api_key_env=LLM_API_KEY_ENV,
            base_url=LLM_BASE_URL,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            temperature=LLM_TEMPERATURE,
        )

    @staticmethod
    def _default_base_url(provider: str) -> str:
        if provider in {"openai_compatible", "gateway"}:
            return "https://api.openai.com/v1"
        if provider == "local":
            return "http://localhost:8000/v1"
        return "https://api.openai.com/v1"

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def api_key(self) -> Optional[str]:
        from config import get_llm_api_key

        return get_llm_api_key()

    def validate_ready(self) -> None:
        if self.provider != "local" and not self.api_key():
            raise LLMClientError(
                f"Missing API key. Set environment variable {self.api_key_env} "
                f"or switch AGENT_BACKEND=rule."
            )

    def complete(self, messages: List[Dict[str, str]]) -> str:
        self.validate_ready()
        if DEBUG_LLM:
            print("MODEL=", self.model)
            print("BASE_URL=", self.base_url)
            print("PROVIDER=", self.provider)
            print("MESSAGES_LEN=", len(messages))
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_chat_completions(self.chat_completions_url, payload)
            except LLMClientError as exc:
                if DEBUG_LLM:
                    traceback.print_exc()
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0 * (attempt + 1), 8.0))
        raise LLMClientError(f"LLM request failed after retries: {last_error}")

    def _post_chat_completions(self, url: str, payload: Dict[str, object]) -> str:
        headers = {"Content-Type": "application/json"}
        api_key = self.api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=context) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMClientError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMClientError(f"Network error calling {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMClientError(f"Timeout calling {url} after {self.timeout}s") from exc
        except Exception as exc:
            raise LLMClientError(f"Unexpected error calling {url}: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Response is not valid JSON: {body[:500]}") from exc
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected LLM response shape: {body[:500]}") from exc
