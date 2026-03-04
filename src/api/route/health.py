from fastapi import APIRouter, Depends, Request
from loguru import logger
from src.api.dependencies import get_workflow
from src.agent.workflow import MultiAgentWorkflow
from src.api.models import HealthCheck

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthCheck)
async def health_check(
    workflow: MultiAgentWorkflow = Depends(get_workflow),
) -> HealthCheck:
    components = {"workflow": "healthy" if workflow else "unhealthy"}
    try:
        if workflow and hasattr(workflow, "app"):
            components["workflow_graph"] = "healthy"
        else:
            components["workflow_graph"] = "not_configured"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        components["workflow"] = "unhealthy"
    all_healthy = all(
        (
            status == "healthy" or status == "not_configured"
            for status in components.values()
        )
    )
    return HealthCheck(
        status="healthy" if all_healthy else "degraded",
        version="2.0.0",
        components=components,
    )


@router.get("/ready")
async def readiness_check(request: Request) -> dict:
    try:
        workflow = request.app.state.workflow
        if workflow is None:
            return {"ready": False, "reason": "Workflow not initialized"}
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get("/live")
async def liveness_check() -> dict:
    return {"alive": True}
