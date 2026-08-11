from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenAICompatibleTextClient:
    """Small dependency-free client for free/open OpenAI-compatible chat APIs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str = "SILICONFLOW_API_KEY",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        opener: Callable | None = None,
    ) -> None:
        self.base_url = base_url.strip()
        self.model = model.strip()
        self.api_key_env = api_key_env.strip()
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.opener = opener or urlopen
        if self.timeout_seconds <= 0:
            raise ValueError("API timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("API max_retries must not be negative")

    @property
    def endpoint(self) -> str:
        if not self.base_url:
            raise ValueError("query parser API base_url is not configured")
        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _content(payload: dict) -> str:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = " ".join(
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        else:
            text = ""
        if not text:
            raise ValueError("API response does not contain assistant text")
        return text

    def generate_text(self, prompt: str, max_new_tokens: int = 384) -> str:
        if not self.model:
            raise ValueError("query parser API model is not configured")
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"missing API key; set environment variable {self.api_key_env}"
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": int(max_new_tokens),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._content(payload)
            except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(
            f"query parser API failed after {self.max_retries + 1} attempts: "
            f"{type(last_error).__name__}"
        ) from last_error
