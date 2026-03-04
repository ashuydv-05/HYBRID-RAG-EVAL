import pytest
from fastapi.testclient import TestClient
from src.api.main import app


class TestAPIImports:
    def test_import_main(self):
        from src.api.main import app

        assert app is not None

    def test_import_models(self):
        from src.api.models import ChatRequest, ChatResponse

        assert ChatRequest is not None
        assert ChatResponse is not None

    def test_import_dependencies(self):
        from src.api.dependencies import get_workflow, get_evaluator

        assert get_workflow is not None
        assert get_evaluator is not None


class TestAPIRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint_exists(self, client):
        response = client.get("/api/health")
        assert response.status_code in [200, 503]

    def test_chat_endpoint_exists(self, client):
        response = client.post("/api/chat", json={"message": "Hello"})
        assert response.status_code in [200, 422, 500]

    def test_evaluation_endpoint_exists(self, client):
        response = client.get("/api/evaluation/health")
        assert response.status_code in [200, 503]


class TestChatRequestModel:
    def test_chat_request_creation(self):
        from src.api.models import ChatRequest

        request = ChatRequest(message="Test query", session_id="test-session")
        assert request.message == "Test query"
        assert request.session_id == "test-session"

    def test_chat_request_optional_session(self):
        from src.api.models import ChatRequest

        request = ChatRequest(message="Test query")
        assert request.message == "Test query"
        assert request.session_id is None


class TestChatResponseModel:
    def test_chat_response_creation(self):
        from src.api.models import ChatResponse, ReasoningStep

        response = ChatResponse(
            answer="Test answer",
            session_id="test-session",
            reasoning_steps=[
                ReasoningStep(thought="Step 1", action="test", observation="")
            ],
            sources=[],
            execution_time=100.0,
        )
        assert response.answer == "Test answer"
        assert response.session_id == "test-session"
        assert len(response.reasoning_steps) == 1
        assert response.execution_time == 100.0
