from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    thought: str
    action: str | None = None
    observation: str | None = None


class Source(BaseModel):
    title: str = "Unknown"
    content: str = ""
    id: int | None = None
    score: float | None = None
    source: str | None = None
    metadata: dict | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(
        None, description="Session ID for conversation continuity"
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID")
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    execution_time: float = Field(..., description="Execution time in milliseconds")


class HealthCheck(BaseModel):
    status: str
    version: str
    components: dict[str, str]
