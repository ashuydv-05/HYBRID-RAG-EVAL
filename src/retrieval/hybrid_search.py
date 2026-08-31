from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

from src.config.clients import get_qdrant_client, get_es_client
from src.config.settings import settings


@dataclass
class SearchResult:
    id: int
    content: str
    title: str
    score: float
    source: str
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalConfig:
    prefetch_k: int = 50
    final_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5


class VectorSearch:
    cross_encoder = None
    embed_model = None

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        es_client: Elasticsearch | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.qdrant = (
            qdrant_client if qdrant_client is not None else get_qdrant_client()
        )
        self.es = es_client if es_client is not None else get_es_client()
        self.config = config or RetrievalConfig()
        self.rrf_k = self.config.rrf_k

    @classmethod
    def get_cross_encoder(cls):
        if cls.cross_encoder is None:
            from sentence_transformers import CrossEncoder

            cls.cross_encoder = CrossEncoder(settings.reranker.model)
        return cls.cross_encoder

    @classmethod
    def get_embed_model(cls):
        if cls.embed_model is None:
            from sentence_transformers import SentenceTransformer

            cls.embed_model = SentenceTransformer(settings.embedding.dense_model)
        return cls.embed_model

    def _to_search_results(self, points, source: str = "dense") -> list[SearchResult]:
        results = []
        for p in points:
            payload = p.payload or {}
            point_id = p.id
            try:
                numeric_id = int(point_id)
            except (TypeError, ValueError):
                numeric_id = abs(hash(str(point_id))) % (10**9)
            results.append(
                SearchResult(
                    id=numeric_id,
                    content=payload.get("content", ""),
                    title=payload.get("title") or payload.get("section") or "Unknown",
                    score=p.score or 0.0,
                    source=source,
                    metadata=payload,
                )
            )
        return results

    def dense_search(self, query: str, top_k: int = 50) -> list[SearchResult]:
        """Query Qdrant with a locally embedded vector.

        Server-side Document inference works from a laptop, but Render often cannot
        use Qdrant Cloud inference. Local MiniLM embeddings match how points were stored.
        """
        collection = settings.qdrant.collection
        dense_name = settings.embedding.dense_vector_name
        query_vector = self.get_embed_model().encode(query).tolist()
        points = self.qdrant.query_points(
            collection_name=collection,
            query=query_vector,
            using=dense_name,
            limit=top_k,
            with_payload=True,
        ).points
        return self._to_search_results(points, source="dense")

    def bm25_search(self, query: str, top_k: int = 50) -> list[SearchResult]:
        if self.es is None:
            return []
        try:
            index = settings.elasticsearch.index
            fields = ["title^3.0", "content^2.0"]
            es_query = {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            }
            response = self.es.search(index=index, query=es_query, size=top_k, request_timeout=1.0)
            results = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                results.append(
                    SearchResult(
                        id=int(hit["_id"]),
                        content=src.get("content", ""),
                        title=src.get("title", "Unknown"),
                        score=float(hit["_score"]),
                        source="bm25",
                        metadata=src,
                    )
                )
            return results
        except Exception as e:
            logger.debug(f"BM25 search skipped: {e}")
            return []

    def rrf_fuse(
        self, results_list: list[list[SearchResult]], top_k: int = 20
    ) -> list[SearchResult]:
        rrf_scores: dict[int, float] = {}
        result_map: dict[int, SearchResult] = {}
        k = self.rrf_k
        for results in results_list:
            for rank, result in enumerate(results, start=1):
                result_map[result.id] = result
                rrf_scores[result.id] = rrf_scores.get(result.id, 0.0) + 1.0 / (
                    k + rank
                )
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [result_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]

    def search(
        self, query: str, top_k: int | None = None, prefetch: int | None = None
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        top_k = top_k or self.config.final_k
        prefetch = prefetch or self.config.prefetch_k
        try:
            dense = []
            try:
                dense = self.dense_search(query, top_k=prefetch)
            except Exception as e:
                logger.warning(f"Dense vector search failed: {e}")

            bm25 = []
            try:
                bm25 = self.bm25_search(query, top_k=prefetch)
            except Exception as e:
                logger.info(f"BM25 Elasticsearch search skipped/unavailable: {e}")

            if dense and bm25:
                rrf_results = self.rrf_fuse([dense, bm25], top_k=self.config.final_k)
            else:
                rrf_results = dense or bm25

            if not rrf_results:
                logger.warning("[VectorSearch] No dense or BM25 hits for query")
                return []

            rerank_enabled = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
            if rerank_enabled:
                return self.rerank(query, rrf_results, top_k)
            return rrf_results[:top_k]
        except Exception as e:
            logger.error(f"[VectorSearch] Error: {e}")
            return []

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        top_k = top_k or self.config.rerank_top_k
        if not results:
            return []
        try:
            reranker = self.get_cross_encoder()
            pairs = [(query, f"{r.title}. {r.content}"[:1000]) for r in results[:10]]
            ce_scores = reranker.predict(pairs)
            scored = list(zip(results[:10], ce_scores))
            scored.sort(key=lambda x: float(x[1]), reverse=True)
            return [r for r, _ in scored[:top_k]]
        except Exception as e:
            logger.warning(f"CrossEncoder unavailable/skipped: {e}")
            return results[:top_k]


class WebSearch:
    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self.client = None
        if self.api_key:
            try:
                from tavily import TavilyClient

                self.client = TavilyClient(api_key=self.api_key)
            except ImportError:
                pass

    def is_available(self) -> bool:
        return self.client is not None

    def search(self, query: str) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if not self.client:
            return []
        try:
            response = self.client.search(query=query, max_results=self.max_results)
            return [
                SearchResult(
                    id=i,
                    content=result.get("content", ""),
                    title=result.get("title", "Web Result"),
                    score=result.get("score", 0.0),
                    source="web",
                    metadata={"url": result.get("url", "")},
                )
                for i, result in enumerate(response.get("results", []))
            ]
        except Exception as e:
            logger.error(f"[WebSearch] Error: {e}")
            return []
