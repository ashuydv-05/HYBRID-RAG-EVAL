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
    chat_history = state.get("chat_history", [])
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
        clarify_answer = "Could you please provide more details about what you're looking for?"
        return {
            "answer": clarify_answer,
            "source": [],
            "reasoning_step": ["GEN: Clarify requested"],
            "chat_history": [
                {
                    "role": "assistant",
                    "content": clarify_answer,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ],
            "node_timings": existing_timings,
        }
    return _gen_rag_answer(llm, query, document, existing_timings, chat_history=chat_history)


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
    llm,
    query: str,
    document: list[dict],
    existing_timings: dict,
    chat_history: list[dict] | None = None,
) -> dict:
    if not document:
        raise NodeInterrupt("Service temporarily unavailable", id="gen")
    context = format_documents(document)

    user_prompt_content = RAG_USER_TEMPLATE.format(context=context, query=query)
    if chat_history and len(chat_history) > 1:
        history_lines = []
        for msg in chat_history[:-1]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        if history_lines:
            conv_str = "\n".join(history_lines[-4:])
            user_prompt_content = f"## Previous Conversation:\n{conv_str}\n\n{user_prompt_content}"

    messages = [
        {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt_content,
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
