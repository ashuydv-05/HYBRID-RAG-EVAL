from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    id: int | str
    question: str
    ground_truth: str
    relevant_documents: list[str] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    id: int | str
    title: str = "Unknown"
    paper_id: str | None = None
    score: float = 0.0
    source: str = "unknown"
    content_snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeScores(BaseModel):
    correctness: float = Field(..., ge=0.0, le=100.0, description="Accuracy vs ground truth (0-100)")
    faithfulness: float = Field(..., ge=0.0, le=100.0, description="Faithfulness to retrieved context (0-100)")
    relevance: float = Field(..., ge=0.0, le=100.0, description="Relevance to user question (0-100)")
    overall: float = Field(..., ge=0.0, le=100.0, description="Weighted composite score (0-100)")
    reason: str = Field(default="", description="Explanation from the judge")


class RetrievalMetrics(BaseModel):
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    annotated: bool = True
    message: str | None = None


class SampleResult(BaseModel):
    question_id: int | str
    retrieval: str  # "vector" | "hybrid"
    llm: str        # "model_1" | "model_2"
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_documents: list[RetrievedDoc] = Field(default_factory=list)
    judge_scores: JudgeScores
    retrieval_metrics: RetrievalMetrics
    execution_time_ms: float
    error: str | None = None


class ConfigSummary(BaseModel):
    retrieval: str
    llm: str
    total_samples: int = 0
    avg_correctness: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_overall: float = 0.0
    avg_precision_at_k: float | None = None
    avg_recall_at_k: float | None = None
    avg_mrr: float | None = None
    avg_latency_ms: float = 0.0


class EvaluationReport(BaseModel):
    timestamp: str
    dataset_path: str
    judge_model: str
    top_k: int
    configurations: dict[str, ConfigSummary]
    best_configuration: str
    best_reason: str
    comparison_matrix: dict[str, dict[str, float]]
    detailed_results: list[SampleResult] = Field(default_factory=list)
