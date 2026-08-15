from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from anima_search.retrieval.openai_compatible import OpenAICompatibleTextClient


class Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_posts_chat_completion():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.get_header("Authorization")
        return Response({
            "choices": [{"message": {"content": '{"query_type":"simple"}'}}]
        })

    client = OpenAICompatibleTextClient(
        "https://example.test/v1",
        "free-model",
        api_key="test-key",
        timeout_seconds=12,
        opener=opener,
    )

    result = client.generate_text("parse this", max_new_tokens=99)

    assert result == '{"query_type":"simple"}'
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "free-model"
    assert captured["body"]["max_tokens"] == 99


def test_openai_compatible_client_retries_transient_failure(monkeypatch):
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError("temporary")
        return Response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("anima_search.retrieval.openai_compatible.time.sleep", lambda _: None)
    client = OpenAICompatibleTextClient(
        "https://example.test/v1",
        "free-model",
        api_key="test-key",
        max_retries=1,
        opener=opener,
    )

    assert client.generate_text("hello") == "ok"
    assert calls == 2


def test_openai_compatible_client_names_missing_key(monkeypatch):
    monkeypatch.delenv("COURSE_FREE_API_KEY", raising=False)
    client = OpenAICompatibleTextClient(
        "https://example.test/v1",
        "free-model",
        api_key_env="COURSE_FREE_API_KEY",
    )
    with pytest.raises(RuntimeError, match="COURSE_FREE_API_KEY"):
        client.generate_text("hello")
