import time
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from src.agent.workflow import MultiAgentWorkflow, WorkflowOutcome
from src.api.dependencies import get_workflow
from src.api.models import ChatRequest, ChatResponse, ReasoningStep, Source

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
    request: ChatRequest, workflow: MultiAgentWorkflow = Depends(get_workflow)
) -> ChatResponse:
    session_id = request.session_id or generate_session_id()
    logger.info(
        f"Chat request - Session: {session_id}, Message: {request.message[:50]}..."
    )
    start_time = time.time()
    try:
        logger.info(f"[API] Starting workflow.run() for session {session_id}")
        result: WorkflowOutcome = workflow.run(request.message, session_id=session_id)
        logger.info(
            f"[API] Workflow completed. Sources count: {len(result.source)}, Reasoning steps: {len(result.reasoning_step)}"
        )
        logger.info(f"[API] Answer preview: {result.answer[:100]}...")
        logger.info(f"[API] Sources: {result.source}")
        execution_time = (time.time() - start_time) * 1000
        return ChatResponse(
            answer=result.answer,
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
