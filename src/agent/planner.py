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


def plan_node(state: AgentState) -> dict:
    import uuid

    call_id = str(uuid.uuid4())[:8]
    logger.info(f"[Planner] Called - id={call_id}, query={state['query'][:50]}...")
    query = state["query"]
    existing_timings = state.get("node_timings", {})
    llm = get_llm_client()
    if not llm:
        raise NodeInterrupt("Service temporarily unavailable", id="planner")

    start_time = time.time()
    try:
        structured_llm = llm.with_structured_output(PlannerOutcome)
        result = structured_llm.invoke(PLANNER_PROMPT_TEMPLATE.invoke({"query": query}))
        elapsed = (time.time() - start_time) * 1000
        new_timings = {**existing_timings, "planner": elapsed}

        if isinstance(result, dict):
            result = PlannerOutcome(**result)

        if result.decision in ["direct_answer", "reject", "clarify"]:
            return {
                "decision": result.decision,
                "route": None,
                "reasoning_step": [
                    f"PLANNER: {result.decision} - routing to generator (id={call_id})"
                ],
                "node_timings": new_timings,
            }
        if result.decision == "process":
            route = result.route or "vector_search"
            return {
                "decision": "process",
                "route": route,
                "reasoning_step": [f"PLANNER: {result.reasoning} - routing to {route}"],
                "node_timings": new_timings,
            }
        return {
            "decision": "process",
            "route": "vector_search",
            "reasoning_step": ["PLANNER: Default routing to vector_search"],
            "node_timings": new_timings,
        }
    except Exception as e:
        logger.error(f"[Planner] Error: {e}")
        raise NodeInterrupt("Service temporarily unavailable", id="planner")
