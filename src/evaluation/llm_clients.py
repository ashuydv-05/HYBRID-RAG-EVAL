from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.prompt import GENERATE_SYSTEM_PROMPT, RAG_USER_TEMPLATE

load_dotenv()


class BaseLLMClient(ABC):
    """Abstract base class for LLM generation clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer given query and context."""
        pass

    @abstractmethod
    def invoke_messages(self, messages: list[dict[str, str]]) -> str:
        """Invoke LLM directly with custom messages."""
        pass


class GroqLLMClient(BaseLLMClient):
    """LLM client implementation using Groq OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        llm: Any = None,
    ):
        model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        super().__init__(model_name=model)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def generate(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RAG_USER_TEMPLATE.format(context=context, query=query),
            },
        ]
        return self.invoke_messages(messages)

    def invoke_messages(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        import time
        from loguru import logger

        langchain_messages = []
        for m in messages:
            if m["role"] == "system":
                langchain_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            else:
                langchain_messages.append(HumanMessage(content=m["content"]))

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(langchain_messages)
                if hasattr(response, "content"):
                    content = response.content
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        return "\n".join([str(item) for item in content])
                    return str(content)
                return str(response)
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Rate limited on {self.model_name}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise
        return ""


class GeminiLLMClient(BaseLLMClient):
    """LLM client implementation for Google Gemini models via OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        llm: Any = None,
    ):
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        super().__init__(model_name=model)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def generate(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RAG_USER_TEMPLATE.format(context=context, query=query),
            },
        ]
        return self.invoke_messages(messages)

    def invoke_messages(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        import time
        from loguru import logger

        langchain_messages = []
        for m in messages:
            if m["role"] == "system":
                langchain_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            else:
                langchain_messages.append(HumanMessage(content=m["content"]))

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(langchain_messages)
                if hasattr(response, "content"):
                    content = response.content
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        return "\n".join([str(item) for item in content])
                    return str(content)
                return str(response)
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Rate limited on Gemini {self.model_name}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        return ""


def get_eval_llm(name: Literal["model_1", "model_2", "gemini"] | str = "model_1") -> BaseLLMClient:
    """Factory to retrieve configured LLM client for model_1, model_2, or Gemini judge."""
    if name == "model_1":
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return GroqLLMClient(model_name=model_name)
    elif name == "model_2":
        model_name = os.getenv("GROQ_MODEL_2", "qwen/qwen3.8-27b")
        return GroqLLMClient(model_name=model_name)
    elif name == "gemini" or name.startswith("gemini-"):
        model_name = name if name.startswith("gemini-") else os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiLLMClient(model_name=model_name)
    elif name == "gpt-4o-mini" or name.startswith("gpt-"):
        openai_key = os.getenv("OPENAI_API_KEY", "")
        return GroqLLMClient(
            model_name=name,
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
        )
    else:
        return GroqLLMClient(model_name=name)
