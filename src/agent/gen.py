from langgraph.errors import NodeInterrupt
from src.config.clients import get_llm_client
from src.agent.state import AgentState
from src.config.prompt import (
    GENERATE_SYSTEM_PROMPT,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
)
from datetime import datetime
import time


def format_documents(document: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(document, 1):
        title = doc.get("title", "Unknown")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {}) or {}
        entry = f"[{i}] {title}\n{content}"
        if paper_id := metadata.get("paper_id"):
            entry += f"\n[paper_id: {paper_id}]"
        elif arxiv_id := metadata.get("arxiv_id"):
            entry += f"\n[arxiv_id: {arxiv_id}]"
        elif url := metadata.get("url"):
            entry += f"\n[url: {url}]"
        parts.append(entry)
    return "\n\n".join(parts)


def gen_node(state: AgentState) -> dict:
    query = state.get("query", "")
    decision = state.get("decision", "process")
    document = state.get("document", [])
    existing_timings = state.get("node_timings", {})
    llm = get_llm_client()
    if not llm:
        raise NodeInterrupt("Service temporarily unavailable", id="gen")
    if decision == "direct_answer":
        return _gen_direct_answer(llm, query, existing_timings)
    if decision == "reject":
        answer = "I cannot help with this request."
        return {
            "answer": answer,
            "source": [],
            "reasoning_step": ["GEN: Rejected"],
            "chat_history": [
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ],
            "node_timings": existing_timings,
        }
    if decision == "clarify":
        return {
            "answer": "Could you please provide more details about what you're looking for?",
            "source": [],
            "reasoning_step": ["GEN: Clarify requested"],
            "chat_history": [],
            "node_timings": existing_timings,
        }
    return _gen_rag_answer(llm, query, document, existing_timings)


def _gen_direct_answer(llm, query: str, existing_timings: dict) -> dict:
    messages = [
        {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = (time.time() - start_time) * 1000
    new_timings = {**existing_timings, "generate": elapsed}

    return {
        "answer": response.content,
        "source": [],
        "reasoning_step": ["GEN: Direct answer"],
        "chat_history": [
            {
                "role": "assistant",
                "content": response.content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ],
        "node_timings": new_timings,
    }


def _gen_rag_answer(
    llm, query: str, document: list[dict], existing_timings: dict
) -> dict:
    if not document:
        raise NodeInterrupt("Service temporarily unavailable", id="gen")
    context = format_documents(document)
    messages = [
        {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RAG_USER_TEMPLATE.format(context=context, query=query),
        },
    ]

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = (time.time() - start_time) * 1000
    new_timings = {**existing_timings, "generate": elapsed}

    return {
        "answer": response.content,
        "source": document,
        "reasoning_step": [f"GEN: RAG answer ({len(document)} source)"],
        "chat_history": [
            {
                "role": "assistant",
                "content": response.content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ],
        "node_timings": new_timings,
    }
