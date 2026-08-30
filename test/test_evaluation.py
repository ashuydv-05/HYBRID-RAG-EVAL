import json
import pytest
from unittest.mock import MagicMock
from src.evaluation.models import (
    EvalQuestion,
    RetrievedDoc,
    JudgeScores,
    RetrievalMetrics,
    ConfigSummary,
    EvaluationReport,
)
from src.evaluation.metrics import compute_retrieval_metrics
from src.evaluation.evaluator import LLMJudge
from src.evaluation.runner import (
    load_dataset,
    format_context,
    convert_search_results,
    EvaluationRunner,
)
from src.retrieval.hybrid_search import SearchResult
from src.retrieval.base import BaseRetriever
from src.evaluation.llm_clients import BaseLLMClient


def test_retrieval_metrics_with_annotations():
    docs = [
        RetrievedDoc(id=1, title="Paper 1", paper_id="1706.03762", score=0.9),
        RetrievedDoc(id=2, title="Paper 2", paper_id="1810.04805", score=0.8),
        RetrievedDoc(id=3, title="Paper 3", paper_id="2005.14165", score=0.7),
        RetrievedDoc(id=4, title="Paper 4", paper_id="1907.11692", score=0.6),
        RetrievedDoc(id=5, title="Paper 5", paper_id="2104.08651", score=0.5),
    ]
    # Ground truth is paper "1706.03762"
    metrics = compute_retrieval_metrics(docs, relevant_ids=["1706.03762"], k=5)
    assert metrics.annotated is True
    assert metrics.precision_at_k == 0.2  # 1 out of 5
    assert metrics.recall_at_k == 1.0     # 1 out of 1
    assert metrics.mrr == 1.0             # Hit at rank 1

    # Ground truth is paper at rank 2
    metrics2 = compute_retrieval_metrics(docs, relevant_ids=["1810.04805"], k=5)
    assert metrics2.mrr == 0.5            # Hit at rank 2 (1/2)


def test_retrieval_metrics_without_annotations():
    docs = [
        RetrievedDoc(id=1, title="Paper 1", paper_id="1706.03762", score=0.9),
    ]
    # No relevant documents annotated
    metrics = compute_retrieval_metrics(docs, relevant_ids=[], k=5)
    assert metrics.annotated is False
    assert metrics.precision_at_k is None
    assert metrics.recall_at_k is None
    assert metrics.mrr is None
    assert "unavailable" in metrics.message


def test_llm_judge_json_parsing():
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.model_name = "test-judge-model"
    mock_client.invoke_messages.return_value = json.dumps({
        "correctness": 88.0,
        "faithfulness": 92.0,
        "relevance": 95.0,
        "overall": 91.3,
        "reason": "Highly accurate and well-grounded."
    })

    judge = LLMJudge(judge_client=mock_client)
    scores = judge.evaluate_sample(
        question="What is BERT?",
        ground_truth="Bidirectional Encoder Representations from Transformers",
        retrieved_context="BERT stands for Bidirectional Encoder Representations from Transformers.",
        generated_answer="BERT is Bidirectional Encoder Representations from Transformers.",
    )

    assert scores.correctness == 88.0
    assert scores.faithfulness == 92.0
    assert scores.relevance == 95.0
    assert scores.overall == 91.3
    assert "accurate" in scores.reason


def test_llm_judge_code_block_parsing():
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.model_name = "test-judge-model"
    mock_client.invoke_messages.return_value = """Here is the score:
```json
{
  "correctness": 80,
  "faithfulness": 85,
  "relevance": 90,
  "overall": 84.5,
  "reason": "Good answer."
}
```
"""

    judge = LLMJudge(judge_client=mock_client)
    scores = judge.evaluate_sample(
        question="What is GPT?",
        ground_truth="Generative Pre-trained Transformer",
        retrieved_context="GPT details",
        generated_answer="GPT is Generative Pre-trained Transformer",
    )
    assert scores.correctness == 80.0
    assert scores.faithfulness == 85.0
    assert scores.relevance == 90.0


def test_load_dataset_file(tmp_path):
    dataset_file = tmp_path / "test_dataset.json"
    data = {
        "questions": [
            {
                "id": 1,
                "question": "What is Transformer?",
                "ground_truth": "Attention-based architecture.",
                "relevant_documents": ["1706.03762"]
            }
        ]
    }
    dataset_file.write_text(json.dumps(data), encoding="utf-8")

    questions = load_dataset(dataset_file)
    assert len(questions) == 1
    assert questions[0].id == 1
    assert questions[0].question == "What is Transformer?"
    assert questions[0].relevant_documents == ["1706.03762"]


def test_runner_2x2_execution(tmp_path):
    # Setup mock retrievers
    mock_vec_retriever = MagicMock(spec=BaseRetriever)
    mock_vec_retriever.search.return_value = [
        SearchResult(id=1, content="Attention doc", title="Attention Is All You Need", score=0.9, source="dense", metadata={"paper_id": "1706.03762"})
    ]

    mock_hyb_retriever = MagicMock(spec=BaseRetriever)
    mock_hyb_retriever.search.return_value = [
        SearchResult(id=1, content="Attention doc", title="Attention Is All You Need", score=0.95, source="hybrid", metadata={"paper_id": "1706.03762"})
    ]

    # Setup mock LLMs
    mock_llm1 = MagicMock(spec=BaseLLMClient)
    mock_llm1.model_name = "mock-llm-1"
    mock_llm1.generate.return_value = "Transformer is based on attention."

    mock_llm2 = MagicMock(spec=BaseLLMClient)
    mock_llm2.model_name = "mock-llm-2"
    mock_llm2.generate.return_value = "Transformer utilizes self-attention layers."

    # Setup mock Judge
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.judge_model_name = "mock-judge"
    mock_judge.evaluate_sample.return_value = JudgeScores(
        correctness=90.0,
        faithfulness=95.0,
        relevance=92.0,
        overall=92.1,
        reason="Excellent answer",
    )

    runner = EvaluationRunner(
        retrievers={"vector": mock_vec_retriever, "hybrid": mock_hyb_retriever},
        llm_clients={"model_1": mock_llm1, "model_2": mock_llm2},
        judge=mock_judge,
        output_dir=tmp_path / "results",
        top_k=1,
    )

    questions = [
        EvalQuestion(
            id=1,
            question="What is the transformer?",
            ground_truth="Attention architecture.",
            relevant_documents=["1706.03762"],
        )
    ]

    report = runner.run_all_combinations(questions=questions)

    # Verify 4 combinations executed
    assert len(report.configurations) == 4
    assert "vector+model_1" in report.configurations
    assert "vector+model_2" in report.configurations
    assert "hybrid+model_1" in report.configurations
    assert "hybrid+model_2" in report.configurations

    assert len(report.detailed_results) == 4
    assert report.comparison_matrix["vector"]["model_1"] == 92.1
    assert report.comparison_matrix["hybrid"]["model_2"] == 92.1

    # Verify files saved
    assert (tmp_path / "results" / "results.json").exists()
    assert (tmp_path / "results" / "summary.json").exists()
