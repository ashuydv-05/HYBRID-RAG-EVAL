import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLangfuseClient:
    def test_langfuse_client_init(self):
        from src.util.langfuse_client import get_langfuse_client

        client = get_langfuse_client()
        assert client is not None
        assert hasattr(client, "is_enabled")
        assert hasattr(client, "create_trace")
        assert hasattr(client, "create_span")
        assert hasattr(client, "add_score")
        assert hasattr(client, "add_feedback")

    def test_langfuse_disabled_without_keys(self, monkeypatch):
        from src.util.langfuse_client import LangfuseClient

        LangfuseClient._instance = None
        LangfuseClient._client = None
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        client = LangfuseClient()
        assert client.is_enabled is False

    def test_langfuse_create_trace_disabled(self, monkeypatch):
        from src.util.langfuse_client import get_langfuse_client

        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        client = get_langfuse_client()
        with client.create_trace(query="test query") as span:
            assert span is None

    def test_langfuse_create_span_disabled(self, monkeypatch):
        from src.util.langfuse_client import get_langfuse_client

        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        client = get_langfuse_client()
        with client.create_span(parent_span=None, name="test_span") as span:
            assert span is None


class TestRAGASEvaluator:
    def test_ragas_evaluator_init(self):
        from src.evaluation.ragas_evaluator import get_ragas_evaluator

        evaluator = get_ragas_evaluator()
        assert evaluator is not None
        assert hasattr(evaluator, "evaluate_single")
        assert hasattr(evaluator, "evaluate_batch")
        assert hasattr(evaluator, "get_aggregated_metrics")
        assert hasattr(evaluator, "is_healthy")

    def test_ragas_evaluate_single_basic(self):
        from src.evaluation.ragas_evaluator import get_ragas_evaluator

        evaluator = get_ragas_evaluator()
        result = evaluator.evaluate_single(
            query="What is machine learning?",
            answer="Machine learning is a field of study that gives computers the ability to learn from data.",
            contexts=[
                "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
            ],
            reference="Machine learning is a subset of AI that enables systems to learn from data.",
        )
        assert result is not None
        assert result.query == "What is machine learning?"
        assert (
            result.answer
            == "Machine learning is a field of study that gives computers the ability to learn from data."
        )
        assert result.scores is not None
        assert "faithfulness" in result.scores
        assert "answer_relevancy" in result.scores
        assert "context_precision" in result.scores
        assert "answer_similarity" in result.scores
        assert result.total_score >= 0.0

    def test_ragas_evaluate_batch(self):
        from src.evaluation.ragas_evaluator import get_ragas_evaluator

        evaluator = get_ragas_evaluator()
        queries = ["What is AI?", "What is deep learning?"]
        answers = [
            "AI is artificial intelligence.",
            "Deep learning is a subset of machine learning.",
        ]
        contexts_list = [
            ["AI refers to the simulation of human intelligence in machines."],
            ["Deep learning is based on artificial neural networks."],
        ]
        references = [
            "AI is artificial intelligence.",
            "Deep learning is a subset of machine learning.",
        ]
        results = evaluator.evaluate_batch(
            queries=queries,
            answers=answers,
            contexts_list=contexts_list,
            references=references,
        )
        assert results is not None
        assert len(results) == 2

    def test_ragas_get_aggregated_metrics(self):
        from src.evaluation.ragas_evaluator import get_ragas_evaluator
        from src.evaluation.ragas_evaluator import RAGASEvaluationResult

        evaluator = get_ragas_evaluator()
        mock_results = [
            RAGASEvaluationResult(
                query="Q1",
                answer="A1",
                contexts=["C1"],
                scores={
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.7,
                    "context_precision": 0.9,
                    "answer_similarity": 0.0,
                },
                total_score=0.75,
            ),
            RAGASEvaluationResult(
                query="Q2",
                answer="A2",
                contexts=["C2"],
                scores={
                    "faithfulness": 0.6,
                    "answer_relevancy": 0.8,
                    "context_precision": 0.7,
                    "answer_similarity": 0.0,
                },
                total_score=0.7,
            ),
        ]
        aggregated = evaluator.get_aggregated_metrics(mock_results)
        assert aggregated is not None
        assert aggregated.faithfulness == 0.7
        assert aggregated.answer_relevancy == 0.75
        assert aggregated.context_precision == 0.8
        assert aggregated.answer_similarity == 0.0
        assert abs(aggregated.total_score - 0.725) < 0.001


class TestBenchmarkQueries:
    def test_get_general_queries(self):
        from src.evaluation.datasets import BenchmarkQueries

        queries = BenchmarkQueries.get_general_queries()
        assert queries is not None
        assert len(queries) == 10
        assert all(("query" in q for q in queries))
        assert all(("expected_topics" in q for q in queries))

    def test_get_advanced_queries(self):
        from src.evaluation.datasets import BenchmarkQueries

        queries = BenchmarkQueries.get_advanced_queries()
        assert queries is not None
        assert len(queries) == 5


class TestDatasetManager:
    def test_create_dataset(self):
        from src.evaluation.datasets import DatasetManager

        manager = DatasetManager(data_dir="/tmp/test_evaluation")
        samples = [
            {"query": "What is AI?", "expected_topics": ["artificial", "intelligence"]},
            {"query": "What is ML?", "expected_topics": ["learning", "data"]},
        ]
        dataset = manager.create_dataset(
            name="test-dataset", description="Test dataset", samples=samples
        )
        assert dataset is not None
        assert dataset.name == "test-dataset"
        assert len(dataset.samples) == 2

    def test_save_and_load_dataset(self):
        from src.evaluation.datasets import DatasetManager

        manager = DatasetManager(data_dir="/tmp/test_evaluation")
        samples = [{"query": "What is AI?", "expected_topics": ["artificial"]}]
        dataset = manager.create_dataset(
            name="save-test-dataset", description="Test save dataset", samples=samples
        )
        filepath = manager.save_dataset(dataset)
        assert filepath == "/tmp/test_evaluation/save-test-dataset.json"
        loaded = manager.load_dataset("save-test-dataset")
        assert loaded is not None
        assert loaded.name == "save-test-dataset"


class TestAPIIntegration:
    def test_app_has_evaluation_router(self):
        from src.api.main import app

        routes = [getattr(r, "path", "") for r in app.routes]
        assert any(("/api/evaluation" in path for path in routes))

    def test_health_endpoint_exists(self):
        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/api/evaluation/health")
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
