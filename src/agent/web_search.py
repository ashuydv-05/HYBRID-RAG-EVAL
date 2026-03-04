from loguru import logger
from langgraph.errors import NodeInterrupt
from src.agent.state import AgentState
from src.retrieval.hybrid_search import WebSearch


def web_search_node(state: AgentState) -> dict:
    query = state.get("query", "")
    try:
        searcher = WebSearch()
        if not searcher.is_available():
            raise NodeInterrupt("Service temporarily unavailable", id="web_search")
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
    except NodeInterrupt:
        raise
    except Exception as e:
        logger.error(f"[WebSearch] Error: {e}")
        raise NodeInterrupt("Service temporarily unavailable", id="web_search")
