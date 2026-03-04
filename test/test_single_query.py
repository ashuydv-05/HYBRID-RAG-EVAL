#!/usr/bin/env python3

import asyncio
import json
import sys
from typing import Any, List

sys.path.insert(0, "/Users/tranthihieu/Documents/Chatbot")

import requests
from llama_index.core.evaluation.retrieval.metrics import Recall, Precision, HitRate
from src.evaluation.ragas_evaluator import RAGASEvaluator

API_BASE_URL = "http://localhost:8000"
API_CHAT = f"{API_BASE_URL}/api/chat"
TIMEOUT = 120


def load_first_query(filepath: str = "data/test_queries.json") -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"][0]


def call_chat_api(query: str) -> dict:
    response = requests.post(API_CHAT, json={"message": query}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def evaluate_retrieval(
    query: str, sources: list[dict], expected_paper_ids: list[str]
) -> dict:
    try:
        retrieved_ids = [
            s.get("metadata", {}).get("paper_id", f"doc_{i}")
            for i, s in enumerate(sources)
        ]

        metrics = {
            "hit_rate": HitRate(),
            "recall@5": Recall(k=5),
            "precision@5": Precision(k=5),
        }

        results = {}
        for name, metric in metrics.items():
            result = metric.compute(
                query=query,
                retrieved_ids=retrieved_ids,
                expected_ids=expected_paper_ids,
            )
            results[name] = result.score

        return results
    except Exception as e:
        print(f"  ⚠️ Retrieval evaluation error: {e}")
        return {
            "hit_rate": None,
            "recall@5": None,
            "precision@5": None,
        }


async def evaluate_generation_async(
    evaluator: RAGASEvaluator,
    query: str,
    answer: str,
    contexts: list[str],
    reference: str | None,
) -> dict[str, Any]:
    eval_result = await evaluator.evaluate_sample(
        query=query,
        answer=answer,
        contexts=contexts,
        reference=reference,
    )
    return {
        "faithfulness": eval_result.scores.get("faithfulness"),
        "answer_relevancy": eval_result.scores.get("answer_relevancy"),
        "context_precision": eval_result.scores.get("context_precision"),
        "answer_similarity": eval_result.scores.get("answer_similarity"),
        "total_score": eval_result.total_score,
    }


def main():
    print("=" * 60)
    print("Quick Test - Single Query Evaluation")
    print("=" * 60)

    query_data = load_first_query()
    query_id = query_data["id"]
    query = query_data["query"]
    expected = query_data.get("expected", {})
    expected_paper_ids = expected.get("paper_ids", [])
    reference_answer = expected.get("answer")

    print(f"\nQuery ID: {query_id}")
    print(f"Query: {query}")
    print(f"Expected papers: {expected_paper_ids}")

    print("\n[1/2] Calling /api/chat...")
    try:
        chat_response = call_chat_api(query)
        answer = chat_response.get("answer", "")
        sources = chat_response.get("sources", [])

        print(f"  ✓ Answer received ({len(answer)} chars)")
        print(f"  ✓ Sources received: {len(sources)}")

        source_paper_ids = [s.get("metadata", {}).get("paper_id", "") for s in sources]
        print(f"  ✓ Retrieved papers: {source_paper_ids[:5]}")

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return

    print("\n[2/2] Running evaluation...")

    print("  - Retrieval (LlamaIndex):")
    retrieval_metrics = evaluate_retrieval(query, sources, expected_paper_ids)

    if retrieval_metrics["hit_rate"] is not None:
        print(f"    ✓ Hit Rate: {retrieval_metrics['hit_rate']:.4f}")
        print(f"    ✓ Recall@5: {retrieval_metrics['recall@5']:.4f}")
        print(f"    ✓ Precision@5: {retrieval_metrics['precision@5']:.4f}")
    else:
        print("    ⚠️ Retrieval evaluation failed")

    print("  - Generation (RAGAS):")
    contexts = [s.get("content", "") for s in sources if s.get("content")]

    try:
        evaluator = RAGASEvaluator()
        eval_result = asyncio.run(
            evaluate_generation_async(
                evaluator=evaluator,
                query=query,
                answer=answer,
                contexts=contexts,
                reference=reference_answer,
            )
        )

        if eval_result["faithfulness"] is not None:
            print(f"    ✓ Faithfulness: {eval_result['faithfulness']:.4f}")
            print(f"    ✓ Answer Relevancy: {eval_result['answer_relevancy']:.4f}")
            if eval_result["context_precision"] is not None:
                print(
                    f"    ✓ Context Precision: {eval_result['context_precision']:.4f}"
                )
            if eval_result["answer_similarity"] is not None:
                print(
                    f"    ✓ Answer Similarity: {eval_result['answer_similarity']:.4f}"
                )
            print(f"    ✓ Total Score: {eval_result['total_score']:.4f}")
        else:
            print("    ⚠️ RAGAS evaluation failed")

    except Exception as e:
        print(f"    ✗ RAGAS Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
