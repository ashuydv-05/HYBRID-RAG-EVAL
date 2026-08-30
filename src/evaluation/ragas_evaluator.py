import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import dotenv_values
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    SemanticSimilarity,
)
from loguru import logger


def get_nvidia_config() -> Dict[str, str]:
    env_vars = dotenv_values(".env")
    return {
        "api_key": env_vars.get("GROQ_API_KEY", ""),
        "base_url": env_vars.get(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ),
        "model": env_vars.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    }


def create_ragas_llm():
    config = get_nvidia_config()
    if not config["api_key"]:
        raise ValueError("GROQ_API_KEY not set")
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    llm = llm_factory(
        config["model"],
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=8192,
    )
    logger.info(f"RAGAS LLM configured: {config['model']} (max_tokens=8192)")
    return llm


def create_ragas_embeddings():
    model = os.getenv("RAGAS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_factory(provider="huggingface", model=model)
    logger.info(f"Embeddings configured: {model}")
    return embeddings


@dataclass
class RAGASEvaluationResult:
    query: str
    answer: str
    contexts: List[str]
    scores: Dict[str, float]
    total_score: float


@dataclass
class EvaluationMetrics:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    answer_similarity: float = 0.0
    total_score: float = 0.0


class RAGASEvaluator:
    def __init__(self):
        self.llm = create_ragas_llm()
        self.embeddings = create_ragas_embeddings()
        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(
            llm=self.llm, embeddings=self.embeddings
        )
        self.context_precision = ContextPrecision(llm=self.llm)
        self.semantic_similarity = SemanticSimilarity(embeddings=self.embeddings)

    async def evaluate_sample(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> RAGASEvaluationResult:
        scores = {}
        faith_result = await self.faithfulness.ascore(
            user_input=query, response=answer, retrieved_contexts=contexts
        )
        scores["faithfulness"] = float(faith_result.value)
        relevancy_result = await self.answer_relevancy.ascore(
            user_input=query, response=answer
        )
        scores["answer_relevancy"] = float(relevancy_result.value)
        if reference:
            precision_result = await self.context_precision.ascore(
                user_input=query, reference=reference, retrieved_contexts=contexts
            )
            scores["context_precision"] = float(precision_result.value)
            similarity_result = await self.semantic_similarity.ascore(
                reference=reference, response=answer
            )
            scores["answer_similarity"] = float(similarity_result.value)
        else:
            scores["context_precision"] = 0.0
            scores["answer_similarity"] = 0.0
        total_score = sum(scores.values()) / len(scores)
        return RAGASEvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            scores=scores,
            total_score=total_score,
        )

    async def evaluate_async(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None,
    ) -> List[RAGASEvaluationResult]:
        tasks = []
        for i in range(len(queries)):
            ref = references[i] if references else None
            tasks.append(
                self.evaluate_sample(
                    query=queries[i],
                    answer=answers[i],
                    contexts=contexts_list[i],
                    reference=ref,
                )
            )
        return await asyncio.gather(*tasks)

    def evaluate(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None,
    ) -> List[RAGASEvaluationResult]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.evaluate_async(queries, answers, contexts_list, references)

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        elif loop:
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> RAGASEvaluationResult:
        return asyncio.run(self.evaluate_sample(query, answer, contexts, reference))

    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None,
    ) -> List[RAGASEvaluationResult]:
        return self.evaluate(
            queries=queries,
            answers=answers,
            contexts_list=contexts_list,
            references=references,
        )

    def get_aggregated_metrics(
        self, results: List[RAGASEvaluationResult]
    ) -> EvaluationMetrics:
        if not results:
            return EvaluationMetrics()
        n = len(results)
        return EvaluationMetrics(
            faithfulness=sum((r.scores["faithfulness"] for r in results)) / n,
            answer_relevancy=sum((r.scores["answer_relevancy"] for r in results)) / n,
            context_precision=sum((r.scores["context_precision"] for r in results)) / n,
            answer_similarity=sum((r.scores["answer_similarity"] for r in results)) / n,
            total_score=sum((r.total_score for r in results)) / n,
        )

    def is_healthy(self) -> bool:
        return self.llm is not None and self.embeddings is not None


def get_ragas_evaluator() -> RAGASEvaluator:
    return RAGASEvaluator()
