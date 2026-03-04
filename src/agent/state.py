from typing import TypedDict, Annotated, Literal, Optional, Any
import operator


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str]


class AgentState(TypedDict):
    query: str
    decision: Literal["direct_answer", "reject", "clarify", "process"]
    route: Optional[Literal["vector_search", "web_search"]]
    document: list[dict]
    validation_result: Optional[Literal["relevant", "insufficient", "off_topic"]]
    answer: str
    source: list[dict]
    reasoning_step: Annotated[list[str], operator.add]
    chat_history: Annotated[list[Message], operator.add]
    node_timings: dict
