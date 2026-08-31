from loguru import logger
from src.agent.state import AgentState
from src.retrieval.hybrid_search import WebSearch


def web_search_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("query", "")
    try:
        searcher = WebSearch()
        if not searcher.is_available():
            logger.warning("[WebSearch] Tavily is not configured; skipping web fallback")
            return {
                "document": [],
                "reasoning_step": ["WEB_SEARCH: Skipped (Tavily not configured)"],
            }
        results = searcher.search(query)
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
            "reasoning_step": [f"WEB_SEARCH: Retrieved {len(docs_as_dicts)} result"],
        }
    except Exception as e:
        logger.error(f"[WebSearch] Error: {e}")
        return {
            "document": [],
            "reasoning_step": [f"WEB_SEARCH: Failed ({e})"],
        }
