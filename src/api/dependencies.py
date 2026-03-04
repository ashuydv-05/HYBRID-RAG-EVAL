from fastapi import Request
from src.agent.workflow import MultiAgentWorkflow


def get_workflow(request: Request) -> MultiAgentWorkflow:
    return request.app.state.workflow
