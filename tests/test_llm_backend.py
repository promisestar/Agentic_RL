"""Tests for user-sim LLM backend (Ollama + OpenAI-compatible / vLLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ecom_rlve.simulator import llm_backend as lb


@pytest.fixture(autouse=True)
def _reset_logger_noise(caplog: pytest.LogCaptureFixture) -> None:
    """Keep test output quiet; individual tests may assert on warnings."""
    yield


def test_llm_generate_ollama_posts_api_chat_with_think(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setattr(lb, "LLM_MODEL", "qwen3.5")
    monkeypatch.setattr(lb, "LLM_TIMEOUT", 5)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "I need running shoes."}}

    with patch.object(lb.requests, "post", return_value=mock_resp) as post:
        text = lb._llm_generate("say hi", seed=7, temperature=0.5, max_tokens=64)

    assert text == "I need running shoes."
    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == "http://ollama.test:11434/api/chat"
    payload = kwargs["json"]
    assert payload["think"] is False
    assert payload["model"] == "qwen3.5"
    assert payload["options"]["seed"] == 7
    assert payload["options"]["temperature"] == 0.5
    assert payload["options"]["num_predict"] == 64


def test_llm_generate_openai_posts_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://vllm.test:8000/v1")
    monkeypatch.setattr(lb, "LLM_MODEL", "Qwen/Qwen3-1.7B")
    monkeypatch.setattr(lb, "LLM_API_KEY", "EMPTY")
    monkeypatch.setattr(lb, "LLM_TIMEOUT", 5)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Looking for a blue jacket."}}],
    }

    with patch.object(lb.requests, "post", return_value=mock_resp) as post:
        text = lb._llm_generate(
            "verbalize",
            seed=1,
            system_prompt="You are a shopper.",
        )

    assert text == "Looking for a blue jacket."
    args, kwargs = post.call_args
    assert args[0] == "http://vllm.test:8000/v1/chat/completions"
    payload = kwargs["json"]
    assert payload["model"] == "Qwen/Qwen3-1.7B"
    assert payload["seed"] == 1
    assert payload["messages"][0]["role"] == "system"
    assert "Authorization" in kwargs["headers"]


def test_llm_generate_openai_appends_v1_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://vllm.test:8000")
    monkeypatch.setattr(lb, "LLM_MODEL", "m")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
    }

    with patch.object(lb.requests, "post", return_value=mock_resp) as post:
        assert lb._llm_generate("x") == "ok"

    assert post.call_args.args[0] == "http://vllm.test:8000/v1/chat/completions"


def test_llm_generate_http_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://vllm.test:8000/v1")

    with patch.object(
        lb.requests,
        "post",
        side_effect=lb.requests.ConnectionError("refused"),
    ):
        assert lb._llm_generate("x") is None


def test_llm_generate_strips_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://ollama.test:11434")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": "<think>secret</think>\nHello there"},
    }

    with patch.object(lb.requests, "post", return_value=mock_resp):
        assert lb._llm_generate("x") == "Hello there"


def test_is_llm_available_ollama_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setattr(lb, "LLM_MODEL", "qwen3.5")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "qwen3.5:latest"}]}

    with patch.object(lb.requests, "get", return_value=mock_resp) as get:
        assert lb.is_llm_available() is True

    assert get.call_args.args[0] == "http://ollama.test:11434/api/tags"


def test_is_llm_available_ollama_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setattr(lb, "LLM_MODEL", "qwen3.5")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "llama3"}]}

    with patch.object(lb.requests, "get", return_value=mock_resp):
        assert lb.is_llm_available() is False


def test_is_llm_available_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://vllm.test:8000/v1")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "Qwen/Qwen3-1.7B"}]}

    with patch.object(lb.requests, "get", return_value=mock_resp) as get:
        assert lb.is_llm_available() is True

    assert get.call_args.args[0] == "http://vllm.test:8000/v1/models"


def test_is_llm_available_openai_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://vllm.test:8000/v1")

    with patch.object(
        lb.requests,
        "get",
        side_effect=lb.requests.ConnectionError("down"),
    ):
        assert lb.is_llm_available() is False


def test_is_ollama_available_aliases_is_llm_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lb, "is_llm_available", lambda: True)
    assert lb.is_ollama_available() is True
    monkeypatch.setattr(lb, "is_llm_available", lambda: False)
    assert lb.is_ollama_available() is False


def test_ollama_generate_alias_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical alias ``_ollama_generate`` must still work."""
    monkeypatch.setattr(lb, "LLM_BACKEND", "openai")
    monkeypatch.setattr(lb, "LLM_BASE_URL", "http://vllm.test:8000/v1")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "alias-ok"}}],
    }

    with patch.object(lb.requests, "post", return_value=mock_resp):
        assert lb._ollama_generate("x") == "alias-ok"
