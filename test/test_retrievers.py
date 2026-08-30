import pytest
from unittest.mock import MagicMock
from src.retrieval.base import BaseRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hybrid_search import SearchResult, VectorSearch


def test_base_retriever_subclass():
    class CustomRetriever(BaseRetriever):
        @property
        def name(self) -> str:
            return "custom"

        def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
            return []

    retriever = CustomRetriever()
    assert retriever.name == "custom"
    assert retriever.search("test") == []


def test_vector_retriever_interface():
    mock_searcher = MagicMock(spec=VectorSearch)
    mock_searcher.dense_search.return_value = [
        SearchResult(id=1, content="Doc 1", title="Title 1", score=0.9, source="dense"),
        SearchResult(id=2, content="Doc 2", title="Title 2", score=0.8, source="dense"),
    ]

    retriever = VectorRetriever(searcher=mock_searcher)
    assert retriever.name == "vector"

    results = retriever.search("attention mechanism", top_k=2)
    assert len(results) == 2
    assert results[0].id == 1
    assert results[0].source == "dense"
    mock_searcher.dense_search.assert_called_once_with(query="attention mechanism", top_k=2)


def test_hybrid_retriever_interface():
    mock_searcher = MagicMock(spec=VectorSearch)
    mock_searcher.search.return_value = [
        SearchResult(id=1, content="Doc 1", title="Title 1", score=0.95, source="dense"),
        SearchResult(id=3, content="Doc 3", title="Title 3", score=0.85, source="bm25"),
    ]

    retriever = HybridRetriever(searcher=mock_searcher)
    assert retriever.name == "hybrid"

    results = retriever.search("transformer models", top_k=2)
    assert len(results) == 2
    assert results[0].id == 1
    mock_searcher.search.assert_called_once_with(query="transformer models", top_k=2)


def test_rrf_rank_fusion_and_deduplication():
    # Test RRF fusion logic directly
    searcher = VectorSearch(qdrant_client=MagicMock(), es_client=MagicMock())
    dense_results = [
        SearchResult(id=1, content="Doc 1", title="Title 1", score=0.9, source="dense"),
        SearchResult(id=2, content="Doc 2", title="Title 2", score=0.8, source="dense"),
        SearchResult(id=3, content="Doc 3", title="Title 3", score=0.7, source="dense"),
    ]
    bm25_results = [
        SearchResult(id=2, content="Doc 2", title="Title 2", score=15.0, source="bm25"),
        SearchResult(id=4, content="Doc 4", title="Title 4", score=12.0, source="bm25"),
        SearchResult(id=1, content="Doc 1", title="Title 1", score=10.0, source="bm25"),
    ]

    fused = searcher.rrf_fuse([dense_results, bm25_results], top_k=3)
    
    # Doc 1 and Doc 2 appear in both lists, so they should be ranked higher due to reciprocal rank summation
    assert len(fused) == 3
    # Check no duplicate doc IDs
    ids = [d.id for d in fused]
    assert len(ids) == len(set(ids))
    # Check doc 1 and 2 are present
    assert 1 in ids
    assert 2 in ids


def test_empty_queries():
    mock_searcher = MagicMock(spec=VectorSearch)
    v_retriever = VectorRetriever(searcher=mock_searcher)
    h_retriever = HybridRetriever(searcher=mock_searcher)

    assert v_retriever.search("") == []
    assert v_retriever.search("   ") == []
    assert h_retriever.search("") == []
    assert h_retriever.search("   ") == []
    mock_searcher.dense_search.assert_not_called()
    mock_searcher.search.assert_not_called()
