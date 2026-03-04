from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from src.config.settings import settings

load_dotenv()


def get_llm_client() -> ChatOpenAI:
    max_tokens = int(os.getenv("NVIDIA_NIM_MAX_TOKENS", "2048"))
    model = os.getenv("NVIDIA_NIM_MODEL", "openai/gpt-oss-120b")
    return ChatOpenAI(
        api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
        base_url=os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        model=model,
        temperature=0.7,
        model_kwargs={"max_tokens": max_tokens},
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)


def get_es_client() -> Elasticsearch:
    return Elasticsearch(settings.elasticsearch.url, request_timeout=60)
