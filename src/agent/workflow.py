from __future__ import annotations
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.errors import NodeInterrupt, GraphRecursionError
from loguru import logger
from src.agent.state import AgentState
from src.agent.planner import plan_node
from src.agent.vector_search import vector_search_node
from src.agent.web_search import web_search_node
from src.agent.validate import validation_node
from src.agent.gen import gen_node
import os
import time
import uuid
from langfuse.langchain import CallbackHandler
from datetime import datetime, timezone
from langgraph.checkpoint.memory import MemorySaver


@dataclass
class WorkflowOutcome:
    answer: str
    source: list[dict] = field(default_factory=list)
    reasoning_step: list[str] = field(default_factory=list)
    node_timings: dict = field(default_factory=dict)
    trace_id: str | None = None
    langfuse_url: str | None = None


def route_question(state: AgentState) -> str:
    decision = state.get("decision")
    route = state.get("route")
    logger.info(f"[Router] Route decision - decision={decision}, route={route}")
    if decision in ["direct_answer", "reject", "clarify"]:
        return "generate"
    if decision == "process":
        if route == "web_search":
            return "web_search"
        return "vector_search"
    logger.warning("[Router] Unknown state, defaulting to generate")
    return "generate"


def decide_to_generate(state: AgentState) -> str:
    validation = state.get("validation_result")
    logger.info(f"[Router] Validation result: {validation}")
    if validation == "relevant":
        return "generate"
    logger.info(f"[Router] Documents {validation}, falling back to web_search")
    return "web_search"


def create_workflow() -> MultiAgentWorkflow:
    return MultiAgentWorkflow()


class MultiAgentWorkflow:
    def __init__(self, checkpointer: MemorySaver | None = None):
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.app = self.create_and_compile_graph()

    def create_and_compile_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", plan_node)
        workflow.add_node("vector_search", vector_search_node)
        workflow.add_node("validate", validation_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("generate", gen_node)
        workflow.set_entry_point("planner")
        workflow.add_conditional_edges(
            "planner",
            route_question,
            {
                "generate": "generate",
                "vector_search": "vector_search",
                "web_search": "web_search",
            },
        )
        workflow.add_edge("vector_search", "validate")
        workflow.add_conditional_edges(
            "validate",
            decide_to_generate,
            {"generate": "generate", "web_search": "web_search"},
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, query: str, session_id: str | None = None) -> WorkflowOutcome:
        thread_id = session_id or str(uuid.uuid4())

        input_state: AgentState = {
            "query": query,
            "search_query": None,
            "decision": "process",
            "route": None,
            "document": [],
            "validation_result": None,
            "answer": "",
            "source": [],
            "reasoning_step": [],
            "chat_history": [
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "node_timings": {},
        }

        callbacks = []
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse.langchain import CallbackHandler
                callbacks.append(CallbackHandler())
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse callback: {e}")

        try:
            result = self.app.invoke(
                input_state,
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 10,
                    "callbacks": callbacks,
                },
            )
        except NodeInterrupt as e:
            logger.error(f"[Workflow] Node interrupted: {e}")
            return WorkflowOutcome(
                answer="Service temporarily unavailable",
                source=[],
                reasoning_step=[f"INTERRUPT: {e}"],
            )
        except GraphRecursionError as e:
            logger.error(f"[Workflow] Recursion limit exceeded: {e}")
            return WorkflowOutcome(
                answer="Service temporarily unavailable",
                source=[],
                reasoning_step=[f"RECURSION_ERROR: {e}"],
            )
        except Exception as e:
            logger.exception(f"[Workflow] Unexpected error: {e}")
            return WorkflowOutcome(
                answer="Service temporarily unavailable",
                source=[],
                reasoning_step=[f"ERROR: {str(e)}"],
            )

        return WorkflowOutcome(
            answer=result.get("answer", ""),
            source=result.get("source", []),
            reasoning_step=result.get("reasoning_step", []),
            node_timings=result.get("node_timings", {}),
        )
