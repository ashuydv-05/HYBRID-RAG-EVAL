import time
from fastapi import APIRouter, Depends, HTTPException, Header
from loguru import logger
from src.agent.workflow import MultiAgentWorkflow, WorkflowOutcome
from src.api.dependencies import get_workflow
from src.api.models import ChatRequest, ChatResponse, ReasoningStep, Source
from src.config.clients import request_groq_api_key, get_llm_client, extract_message_text

router = APIRouter(prefix="/chat", tags=["chat"])
STEP_MAP = {
    "PLANNER:": ("planner", "Query analyzed"),
    "VECTOR_SEARCH:": ("vector_search", "Searching arXiv papers"),
    "WEB_SEARCH:": ("web_search", "Searching the web"),
    "VALIDATE:": ("validate", "Document validated"),
    "GEN:": ("generate", "Answer generated"),
    "ERROR:": ("error", "Error occurred"),
}


def generate_session_id() -> str:
    import uuid

    return str(uuid.uuid4())


def convert_sources(sources: list[dict]) -> list[Source]:
    return [Source.model_validate(s) for s in sources]


def build_reasoning_steps(result: WorkflowOutcome) -> list[ReasoningStep]:
    steps = []
    seen_steps = set()
    for step_text in result.reasoning_step:
        if step_text in seen_steps:
            continue
        seen_steps.add(step_text)
        for prefix, (action, default_obs) in STEP_MAP.items():
            if step_text.startswith(prefix):
                obs = default_obs
                if action == "retrieve":
                    obs = step_text
                steps.append(
                    ReasoningStep(thought=step_text, action=action, observation=obs)
                )
                break
        else:
            steps.append(
                ReasoningStep(thought=step_text, action="step", observation="")
            )
    return steps


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    workflow: MultiAgentWorkflow = Depends(get_workflow),
    x_groq_api_key: str | None = Header(None, alias="x-groq-api-key"),
) -> ChatResponse:
    header_key = (x_groq_api_key or request.groq_api_key or "").strip()
    token = request_groq_api_key.set(header_key or None)

    session_id = request.session_id or generate_session_id()
    logger.info(
        f"Chat request - Session: {session_id}, Message: {request.message[:50]}..."
    )
    if workflow is None:
        workflow = MultiAgentWorkflow()
    start_time = time.time()
    try:
        logger.info(f"[API] Starting workflow.run() for session {session_id}")
        result: WorkflowOutcome = workflow.run(request.message, session_id=session_id)
        logger.info(
            f"[API] Workflow completed. Sources count: {len(result.source)}, Reasoning steps: {len(result.reasoning_step)}"
        )
        answer = result.answer
        if not answer or not str(answer).strip():
            logger.warning("[API] Workflow returned empty answer, generating fallback response...")
            try:
                llm = get_llm_client()
                direct_resp = llm.invoke(
                    f"You are an expert AI academic research assistant. Please answer this query thoroughly: {request.message}"
                )
                answer = extract_message_text(direct_resp)
            except Exception as ex:
                logger.error(f"[API] Fallback generation error: {ex}")
                answer = ""
            if not answer:
                steps = "; ".join(result.reasoning_step[-3:]) if result.reasoning_step else "no agent steps"
                answer = (
                    "I could not complete retrieval-backed generation for this question. "
                    f"Last agent steps: {steps}. Check Qdrant connectivity and Groq model output."
                )

        execution_time = (time.time() - start_time) * 1000
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            reasoning_steps=build_reasoning_steps(result),
            sources=convert_sources(result.source),
            execution_time=execution_time,
            node_timings=result.node_timings,
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process request: {str(e)}"
        )
    finally:
        request_groq_api_key.reset(token)
