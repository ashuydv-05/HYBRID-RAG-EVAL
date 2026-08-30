from loguru import logger
from src.retrieval.base import BaseRetriever
from src.retrieval.hybrid_search import SearchResult, VectorSearch


class VectorRetriever(BaseRetriever):
    """Semantic vector-only retriever using Qdrant dense embeddings."""

    def __init__(self, searcher: VectorSearch | None = None):
        self.searcher = searcher if searcher is not None else VectorSearch()

    @property
    def name(self) -> str:
        return "vector"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        try:
            return self.searcher.dense_search(query=query, top_k=top_k)
        except Exception as e:
            logger.error(f"[VectorRetriever] Search failed: {e}")
            return []
