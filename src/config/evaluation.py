import os
from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sample_rate: float = 0.1
    batch_size: int = 20
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        return cls(
            llm_model=os.getenv("RAGAS_LLM_MODEL", "openai/gpt-oss-120b"),
            embedding_model=os.getenv(
                "RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            sample_rate=float(os.getenv("EVAL_SAMPLE_RATE", "0.1")),
            batch_size=int(os.getenv("EVAL_BATCH_SIZE", "20")),
            enabled=os.getenv("EVAL_ENABLED", "true").lower() == "true",
        )


@dataclass
class RAGMetricsConfig:
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    context_precision_threshold: float = 0.6

    @classmethod
    def from_env(cls) -> "RAGMetricsConfig":
        return cls(
            faithfulness_threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7")),
            answer_relevancy_threshold=float(
                os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.7")
            ),
            context_precision_threshold=float(
                os.getenv("CONTEXT_PRECISION_THRESHOLD", "0.6")
            ),
        )
