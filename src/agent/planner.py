from typing import Literal, Optional
from loguru import logger
from pydantic import BaseModel
from langgraph.errors import NodeInterrupt
from src.config.clients import get_llm_client
from src.agent.state import AgentState
from src.config.prompt import PLANNER_PROMPT_TEMPLATE
import time


class PlannerOutcome(BaseModel):
    decision: Literal["direct_answer", "reject", "clarify", "process"]
    route: Optional[Literal["vector_search", "web_search"]] = None
    reasoning: str
    search_query: Optional[str] = None


def format_chat_history(chat_history: list[dict], current_query: str) -> str:
    """Format past messages for the planner prompt."""
    if not chat_history:
        return "(No previous messages in this session)"

    # Filter out current message if already in chat_history
    history_to_format = chat_history[:-1] if (
        chat_history and chat_history[-1].get("role") == "user" and chat_history[-1].get("content") == current_query
    ) else chat_history

    if not history_to_format:
        return "(No previous messages in this session)"

    lines = []
    for msg in history_to_format[-6:]:  # Keep last 3 turns
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def plan_node(state: AgentState) -> dict:
    import uuid

    call_id = str(uuid.uuid4())[:8]
    query = state["query"]
    logger.info(f"[Planner] Called - id={call_id}, query={query[:50]}...")
    existing_timings = state.get("node_timings", {})
    llm = get_llm_client()
    if not llm:
        raise NodeInterrupt("Service temporarily unavailable", id="planner")

    history_str = format_chat_history(state.get("chat_history", []), query)

    start_time = time.time()
    try:
        structured_llm = llm.with_structured_output(PlannerOutcome)
        result = structured_llm.invoke(
            PLANNER_PROMPT_TEMPLATE.invoke({"query": query, "chat_history": history_str})
        )
        elapsed = (time.time() - start_time) * 1000
        new_timings = {**existing_timings, "planner": elapsed}

        if isinstance(result, dict):
            result = PlannerOutcome(**result)

        resolved_search_query = result.search_query or query

        if result.decision in ["direct_answer", "reject", "clarify"]:
            return {
                "decision": result.decision,
                "route": None,
                "search_query": resolved_search_query,
                "reasoning_step": [
                    f"PLANNER: {result.decision} - {result.reasoning} (id={call_id})"
                ],
                "node_timings": new_timings,
            }
        if result.decision == "process":
            route = result.route or "vector_search"
            return {
                "decision": "process",
                "route": route,
                "search_query": resolved_search_query,
                "reasoning_step": [f"PLANNER: {result.reasoning} - resolved search query: \"{resolved_search_query}\" - routing to {route}"],
                "node_timings": new_timings,
            }
        return {
            "decision": "process",
            "route": "vector_search",
            "search_query": query,
            "reasoning_step": ["PLANNER: Default routing to vector_search"],
            "node_timings": new_timings,
        }
    except Exception as e:
        logger.error(f"[Planner] Error: {e}")
        raise NodeInterrupt("Service temporarily unavailable", id="planner")
