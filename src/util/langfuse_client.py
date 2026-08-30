import os
from contextlib import contextmanager
from typing import Any, Generator, Optional
from dotenv import load_dotenv

load_dotenv()


class LangfuseClient:
    _instance: Optional["LangfuseClient"] = None
    _client: Any = None

    def __init__(self):
        enabled_env = os.getenv("LANGFUSE_ENABLED", "true").lower()
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.is_enabled = enabled_env == "true" and bool(public_key and secret_key)

    @contextmanager
    def create_trace(self, query: str, **kwargs) -> Generator[Optional[Any], None, None]:
        if not self.is_enabled:
            yield None
        else:
            yield None

    @contextmanager
    def create_span(self, parent_span: Any = None, name: str = "span", **kwargs) -> Generator[Optional[Any], None, None]:
        if not self.is_enabled:
            yield None
        else:
            yield None

    def add_score(self, name: str, value: float, **kwargs) -> None:
        pass

    def add_feedback(self, score: float, comment: Optional[str] = None, **kwargs) -> None:
        pass


def get_langfuse_client() -> LangfuseClient:
    return LangfuseClient()
