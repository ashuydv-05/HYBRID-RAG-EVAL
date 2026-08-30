from src.retrieval.base import BaseRetriever
from src.retrieval.hybrid_search import SearchResult, VectorSearch


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic vector search and BM25 keyword search with RRF."""

    def __init__(self, searcher: VectorSearch | None = None):
        self.searcher = searcher if searcher is not None else VectorSearch()

    @property
    def name(self) -> str:
        return "hybrid"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        return self.searcher.search(query=query, top_k=top_k)
