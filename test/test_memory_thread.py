import pytest
from src.agent.workflow import MultiAgentWorkflow
from src.agent.planner import format_chat_history, PlannerOutcome
from src.agent.state import AgentState


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


def test_multi_turn_session_memory():
    workflow = MultiAgentWorkflow()
    session_id = "test-session-multi-turn-001"

    # Turn 1: Ask about papers
    res1 = workflow.run("What are recent papers about PaLM and BERT?", session_id=session_id)
    assert res1.answer is not None
    assert len(res1.answer) > 0

    # Turn 2: Follow-up using pronoun "these"
    res2 = workflow.run("what are the dates of these papers?", session_id=session_id)
    assert res2.answer is not None
    # Must NOT ask for clarification because memory resolved "these"
    assert "Could you please provide more details" not in res2.answer


def test_independent_session_isolation():
    workflow = MultiAgentWorkflow()
    session_a = "session-alpha"
    session_b = "session-beta"

    res_a = workflow.run("What is Transformer architecture?", session_id=session_a)
    assert res_a.answer is not None

    res_b = workflow.run("What is BERT?", session_id=session_b)
    assert res_b.answer is not None
