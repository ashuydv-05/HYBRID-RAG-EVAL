from typing import Literal
from loguru import logger
from pydantic import BaseModel
from src.config.clients import get_llm_client
from src.agent.state import AgentState
from src.config.prompt import VALIDATE_SYSTEM_PROMPT, VALIDATE_HUMAN_TEMPLATE
import time


class ValidationOutcome(BaseModel):
    validation: Literal["relevant", "insufficient", "off_topic"]
    reasoning: str


def validation_node(state: AgentState) -> dict:
    query = state.get("query", "")
    document = state.get("document", [])
    existing_timings = state.get("node_timings", {})

    if not document:
        logger.info("[Validate] No documents to validate")
        return {
            "document": document,
            "validation_result": "insufficient",
            "reasoning_step": ["VALIDATE: No documents found"],
            "node_timings": existing_timings,
        }
    llm = get_llm_client()
    if not llm:
        logger.warning("[Validate] No LLM available, assuming relevant")
        return {
            "document": document,
            "validation_result": "relevant",
            "reasoning_step": ["VALIDATE: No LLM, assuming relevant"],
            "node_timings": existing_timings,
        }
    docs_text = "\n".join(
        (
            f"[{i + 1}] {d.get('title', 'Unknown')}: {d.get('content', '')[:500]}"
            for i, d in enumerate(document[:3])
        )
    )
    messages = [
        {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": VALIDATE_HUMAN_TEMPLATE.format(query=query, documents=docs_text),
        },
    ]

    start_time = time.time()
    try:
        structured_llm = llm.with_structured_output(ValidationOutcome)
        result = structured_llm.invoke(messages)
        elapsed = (time.time() - start_time) * 1000
        new_timings = {**existing_timings, "validate": elapsed}

        if isinstance(result, dict):
            result = ValidationOutcome(**result)

        logger.info(f"[Validate] Result: {result.validation} - {result.reasoning}")
        return {
            "document": document,
            "validation_result": result.validation,
            "reasoning_step": [f"VALIDATE: {result.validation} - {result.reasoning}"],
            "node_timings": new_timings,
        }
    except Exception as e:
        logger.error(f"[Validate] Error: {e}")
        return {
            "document": document,
            "validation_result": "relevant" if document else "insufficient",
            "reasoning_step": [
                f"VALIDATE: Assumed {'relevant' if document else 'insufficient'} after grader error ({e})"
            ],
            "node_timings": {**existing_timings, "validate": (time.time() - start_time) * 1000},
        }
