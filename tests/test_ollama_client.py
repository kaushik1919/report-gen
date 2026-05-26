"""
Tests for core/ollama_client.py.
All network calls are mocked — no Ollama daemon required.
"""
from unittest.mock import MagicMock, patch

import pytest

from core.ollama_client import (
    OllamaClientError,
    check_connectivity,
    generate,
    list_models,
    model_exists,
)


def _chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    @patch("core.ollama_client.ollama.Client")
    def test_returns_content_string(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_response('{"key": "value"}')
        result = generate("test prompt", model="llama3.1:8b")
        assert result == '{"key": "value"}'

    @patch("core.ollama_client.ollama.Client")
    def test_passes_json_format(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_response("{}")
        generate("prompt", model="llama3.1:8b")
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        assert call_kwargs.get("format") == "json"

    @patch("core.ollama_client.ollama.Client")
    def test_passes_temperature_in_options(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_response("{}")
        generate("prompt", model="llama3.1:8b", temperature=0.7)
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.7

    @patch("core.ollama_client.ollama.Client")
    def test_connection_error_raises_client_error(self, MockClient):
        MockClient.return_value.chat.side_effect = ConnectionRefusedError("refused")
        with pytest.raises(OllamaClientError, match="Ollama request failed"):
            generate("prompt", model="llama3.1:8b")

    @patch("core.ollama_client.ollama.Client")
    def test_timeout_error_raises_client_error(self, MockClient):
        MockClient.return_value.chat.side_effect = TimeoutError("timed out")
        with pytest.raises(OllamaClientError):
            generate("prompt", model="llama3.1:8b")

    @patch("core.ollama_client.ollama.Client")
    def test_prompt_included_in_messages(self, MockClient):
        MockClient.return_value.chat.return_value = _chat_response("{}")
        generate("my test prompt", model="llama3.1:8b")
        call_kwargs = MockClient.return_value.chat.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "my test prompt"
        assert call_kwargs["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# check_connectivity()
# ---------------------------------------------------------------------------

class TestCheckConnectivity:
    @patch("core.ollama_client.ollama.Client")
    def test_returns_true_when_server_answers(self, MockClient):
        MockClient.return_value.list.return_value = MagicMock()
        assert check_connectivity() is True

    @patch("core.ollama_client.ollama.Client")
    def test_returns_false_on_connection_error(self, MockClient):
        MockClient.return_value.list.side_effect = ConnectionRefusedError("refused")
        assert check_connectivity() is False

    @patch("core.ollama_client.ollama.Client")
    def test_returns_false_on_timeout(self, MockClient):
        MockClient.return_value.list.side_effect = TimeoutError("timeout")
        assert check_connectivity() is False


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------

class TestListModels:
    @patch("core.ollama_client.ollama.Client")
    def test_returns_model_name_strings(self, MockClient):
        m1 = MagicMock()
        m1.model = "llama3.1:8b"
        m2 = MagicMock()
        m2.model = "qwen2.5:7b"
        result_mock = MagicMock()
        result_mock.models = [m1, m2]
        MockClient.return_value.list.return_value = result_mock
        models = list_models()
        assert models == ["llama3.1:8b", "qwen2.5:7b"]

    @patch("core.ollama_client.ollama.Client")
    def test_returns_empty_list_on_error(self, MockClient):
        MockClient.return_value.list.side_effect = Exception("unreachable")
        assert list_models() == []

    @patch("core.ollama_client.ollama.Client")
    def test_returns_empty_list_when_no_models(self, MockClient):
        result_mock = MagicMock()
        result_mock.models = []
        MockClient.return_value.list.return_value = result_mock
        assert list_models() == []


# ---------------------------------------------------------------------------
# model_exists()
# ---------------------------------------------------------------------------

class TestModelExists:
    @patch("core.ollama_client.ollama.Client")
    def test_returns_true_when_model_present(self, MockClient):
        m = MagicMock()
        m.model = "llama3.1:8b"
        result_mock = MagicMock()
        result_mock.models = [m]
        MockClient.return_value.list.return_value = result_mock
        assert model_exists("llama3.1:8b") is True

    @patch("core.ollama_client.ollama.Client")
    def test_returns_false_when_model_absent(self, MockClient):
        result_mock = MagicMock()
        result_mock.models = []
        MockClient.return_value.list.return_value = result_mock
        assert model_exists("nonexistent:7b") is False
