#!/usr/bin/env python3

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, List

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
from tqdm import tqdm
from src.evaluation.ragas_evaluator import RAGASEvaluator

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_CHAT = f"{API_BASE_URL}/api/chat"
TIMEOUT = 500
MAX_CONCURRENT = 5


def load_test_queries(
    filepath: str = "data/test/test_queries.json",
) -> list[dict[str, Any]]:
    filepath = str(PROJECT_ROOT / filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def call_chat_api(query: str, session_id: str | None = None) -> dict[str, Any]:
    payload = {"message": query}
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(API_CHAT, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


async def evaluate_sample_async(
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


def evaluate_query_simple(query_data: dict[str, Any]) -> dict[str, Any]:
    """Wrapper for concurrent execution without progress bar."""
    result = {
        "query_id": query_data["id"],
        "query": query_data["query"],
        "expected_paper_ids": query_data.get("expected", {}).get("paper_ids", []),
        "ragas_metrics": {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "answer_similarity": None,
            "total_score": None,
        },
        "api_response": None,
        "error": None,
    }

    query = query_data["query"]
    expected = query_data.get("expected", {})
    expected_paper_ids = expected.get("paper_ids", [])
    reference_answer = expected.get("answer")

    try:
        chat_response = call_chat_api(query, session_id=f"eval-{query_data['id']}")
        result["api_response"] = {
            "answer": chat_response.get("answer", ""),
            "sources_count": len(chat_response.get("sources", [])),
        }

        sources = chat_response.get("sources", [])

        contexts = [s.get("content", "") for s in sources if s.get("content")]
        if contexts and chat_response.get("answer"):
            try:
                evaluator = RAGASEvaluator()
                eval_result = asyncio.run(
                    evaluate_sample_async(
                        evaluator=evaluator,
                        query=query,
                        answer=chat_response["answer"],
                        contexts=contexts,
                        reference=reference_answer,
                    )
                )
                result["ragas_metrics"] = eval_result
            except Exception as e:
                result["error"] = f"Evaluation error: {str(e)}"

    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)

    ragas_metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "answer_similarity",
        "total_score",
    ]

    errors = [r["error"] for r in results if r["error"]]
    timeout_count = sum(1 for e in errors if e == "Timeout")
    error_count = len(errors)

    agg_ragas = {}
    for metric in ragas_metrics:
        values = [
            r["ragas_metrics"].get(metric)
            for r in results
            if r["ragas_metrics"].get(metric) is not None
        ]
        agg_ragas[metric] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "count": len(values),
        }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_queries": total,
        },
        "ragas_metrics": agg_ragas,
        "errors": {
            "total": error_count,
            "timeout": timeout_count,
            "details": errors[:10],
        },
    }


def main():
    print("=" * 60)
    print("arXiv Research Assistant - Evaluation Script")
    print("=" * 60)

    print("\n[1/5] Loading test queries...")
    queries = load_test_queries()
    print(f"    Loaded {len(queries)} queries from data/test/test_queries.json")

    print(f"\n[2/5] Checking API connectivity...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            print(f"    ✓ API is healthy at {API_BASE_URL}")
        else:
            print(f"    ⚠ API returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Cannot connect to API: {e}")
        print(f"    Make sure the backend is running at {API_BASE_URL}")
        sys.exit(1)

    print(f"\n[3/5] Running evaluation...")
    print(f"    Total queries: {len(queries)}")
    print(f"    Max concurrent: {MAX_CONCURRENT} requests")
    print(f"    Timeout per request: {TIMEOUT}s")
    print(
        f"    Estimated time: ~{len(queries) * TIMEOUT / MAX_CONCURRENT / 60:.1f} minutes"
    )
    print("-" * 60)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(evaluate_query_simple, q): q for q in queries}
        with tqdm(total=len(queries), desc="Evaluating", unit="query") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    query_data = futures[future]
                    results.append(
                        {
                            "query_id": query_data["id"],
                            "query": query_data["query"],
                            "error": f"Future error: {str(e)}",
                        }
                    )
                pbar.update(1)

    print("-" * 60)

    print(f"\n[4/5] Aggregating results...")
    summary = aggregate_results(results)

    print(f"\n[5/5] Saving results...")
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Saved detailed results to {results_path}")

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Total queries: {summary['summary']['total_queries']}")

    print(f"\n🎯 RAGAS Metrics:")
    for metric, data in summary["ragas_metrics"].items():
        if data["mean"] is not None:
            print(f"   {metric}: {data['mean']:.4f} (n={data['count']})")

    if summary["errors"]["total"] > 0:
        print(
            f"\n⚠️  Errors: {summary['errors']['total']} ({summary['errors']['timeout']} timeouts)"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
