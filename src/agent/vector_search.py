from loguru import logger
from langgraph.errors import NodeInterrupt
from src.agent.state import AgentState
from src.retrieval.hybrid_search import VectorSearch
import time


def vector_search_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("query", "")
    existing_timings = state.get("node_timings", {})

    start_time = time.time()
    try:
        searcher = VectorSearch()
        results = searcher.search(query, top_k=5)
        elapsed = (time.time() - start_time) * 1000
        new_timings = {**existing_timings, "vector_search": elapsed}

        docs_as_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "title": doc.title,
                "score": doc.score,
                "source": doc.source,
                "metadata": doc.metadata,
            }
            for doc in results
        ]

        return {
            "document": docs_as_dicts,
            "reasoning_step": state.get("reasoning_step", [])
            + [
                f"VECTOR_SEARCH: Retrieved {len(docs_as_dicts)} document for: {query[:50]}..."
            ],
            "node_timings": new_timings,
        }
    except Exception as e:
        logger.error(f"[VectorSearch] Error: {e}")
        raise NodeInterrupt("Service temporarily unavailable", id="vector_search")
