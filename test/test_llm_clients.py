import pytest
from unittest.mock import MagicMock
from src.evaluation.llm_clients import BaseLLMClient, GroqLLMClient, get_eval_llm


def test_base_llm_client_interface():
    class CustomLLM(BaseLLMClient):
        def generate(self, query: str, context: str) -> str:
            return f"Answer for {query} with {context}"

        def invoke_messages(self, messages: list[dict[str, str]]) -> str:
            return "ok"

    client = CustomLLM(model_name="test-model")
    assert client.model_name == "test-model"
    assert client.generate("q", "c") == "Answer for q with c"


def test_groq_llm_client_mock():
    mock_langchain_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Transformers use self-attention."
    mock_langchain_llm.invoke.return_value = mock_response

    client = GroqLLMClient(
        model_name="test-llm",
        api_key="mock_key",
        llm=mock_langchain_llm,
    )

    assert client.model_name == "test-llm"
    res = client.generate(query="What is transformer?", context="[1] Paper\nAttention details")
    assert res == "Transformers use self-attention."
    mock_langchain_llm.invoke.assert_called_once()


def test_get_eval_llm_factory(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "custom-model-1")
    monkeypatch.setenv("GROQ_MODEL_2", "custom-model-2")

    client1 = get_eval_llm("model_1")
    assert client1.model_name == "custom-model-1"

    client2 = get_eval_llm("model_2")
    assert client2.model_name == "custom-model-2"

    client_custom = get_eval_llm("llama-3-8b")
    assert client_custom.model_name == "llama-3-8b"
