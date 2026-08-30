from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from src.config.settings import settings

load_dotenv()


def get_llm_client() -> ChatOpenAI:
    max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY", ""),
        base_url=os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ),
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
    )


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", settings.qdrant.url)
    api_key = os.getenv("QDRANT_API_KEY", settings.qdrant.api_key)
    return QdrantClient(
        url=url,
        api_key=api_key if api_key else None,
        check_compatibility=False,
    )


def get_es_client() -> Elasticsearch:
    return Elasticsearch(settings.elasticsearch.url, request_timeout=60)
