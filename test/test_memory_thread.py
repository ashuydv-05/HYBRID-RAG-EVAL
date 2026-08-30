import pytest
from unittest.mock import MagicMock, patch
from src.agent.planner import format_chat_history, PlannerOutcome, plan_node
from src.agent.state import AgentState
from src.agent.workflow import create_workflow, MultiAgentWorkflow


def test_format_chat_history_empty():
    assert format_chat_history([], "test") == "(No previous messages in this session)"


def test_format_chat_history_with_turns():
    history = [
        {"role": "user", "content": "What is PaLM?"},
        {"role": "assistant", "content": "PaLM is a 540B parameter language model from Google."},
        {"role": "user", "content": "When was it published?"},
    ]
    formatted = format_chat_history(history, "When was it published?")
    assert "User: What is PaLM?" in formatted
    assert "Assistant: PaLM is a 540B parameter language model" in formatted
    # The current message should not be duplicated in the history portion
    assert formatted.count("When was it published?") == 0


def test_plan_node_contextual_query_rewriting():
    # Mock LLM to return rewritten query when pronoun is present
    mock_outcome = PlannerOutcome(
        decision="process",
        route="vector_search",
        reasoning="User refers to PaLM and BERT papers from previous context",
        search_query="PaLM and BERT publication dates arXiv",
    )

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_outcome
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("src.agent.planner.get_llm_client", return_value=mock_llm):
        state = {
            "query": "what are the dates of these papers?",
            "chat_history": [
                {"role": "user", "content": "Tell me about PaLM and BERT"},
                {"role": "assistant", "content": "PaLM and BERT are transformer models."},
            ],
            "messages": [],
            "reasoning_steps": [],
            "sources": [],
            "node_timings": {},
        }
        res = plan_node(state)
        assert res["decision"] == "process"
        assert res["route"] == "vector_search"
        assert res["search_query"] == "PaLM and BERT publication dates arXiv"


def test_workflow_thread_checkpointer():
    workflow = create_workflow()
    assert hasattr(workflow, "app")
    assert workflow.app is not None
    # Verify app has checkpointer attached for session memory
    assert hasattr(workflow.app, "checkpointer")
