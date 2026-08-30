from fastapi import Request
from src.agent.workflow import MultiAgentWorkflow


def get_workflow(request: Request) -> MultiAgentWorkflow | None:
    return getattr(request.app.state, "workflow", None)


def get_evaluator():
    from src.evaluation.ragas_evaluator import get_ragas_evaluator

    return get_ragas_evaluator()
