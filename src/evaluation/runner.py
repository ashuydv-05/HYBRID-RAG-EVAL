from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from loguru import logger

from src.retrieval.base import BaseRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hybrid_search import SearchResult
from src.evaluation.llm_clients import BaseLLMClient, get_eval_llm
from src.evaluation.evaluator import LLMJudge
from src.evaluation.metrics import compute_retrieval_metrics
from src.evaluation.models import (
    EvalQuestion,
    RetrievedDoc,
    SampleResult,
    ConfigSummary,
    EvaluationReport,
)


def load_dataset(dataset_path: str | Path) -> list[EvalQuestion]:
    """Load evaluation questions from JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either {"questions": [...]} or raw list [...]
    items = data.get("questions", data) if isinstance(data, dict) else data
    questions = []
    for item in items:
        # Handle format with expected.answer or ground_truth
        qid = item.get("id", len(questions) + 1)
        q_text = item.get("question") or item.get("query", "")
        ground_truth = item.get("ground_truth") or item.get("expected", {}).get("answer", "")
        rel_docs = item.get("relevant_documents") or item.get("expected", {}).get("paper_ids", [])
        if q_text:
            questions.append(
                EvalQuestion(
                    id=qid,
                    question=q_text,
                    ground_truth=ground_truth,
                    relevant_documents=[str(p) for p in rel_docs],
                )
            )
    return questions


def format_context(results: Sequence[SearchResult]) -> str:
    """Format retrieved documents into a context block."""
    parts = []
    for i, doc in enumerate(results, 1):
        title = doc.title or "Unknown"
        content = doc.content or ""
        meta = doc.metadata or {}
        paper_id = meta.get("paper_id") or meta.get("arxiv_id") or ""
        pid_str = f" [paper_id: {paper_id}]" if paper_id else ""
        parts.append(f"[{i}] {title}{pid_str}\n{content}")
    return "\n\n".join(parts)


def convert_search_results(results: Sequence[SearchResult]) -> list[RetrievedDoc]:
    """Convert SearchResult domain objects into RetrievedDoc serializable models."""
    docs = []
    for r in results:
        meta = r.metadata or {}
        paper_id = meta.get("paper_id") or meta.get("arxiv_id")
        docs.append(
            RetrievedDoc(
                id=r.id,
                title=r.title or "Unknown",
                paper_id=str(paper_id) if paper_id else None,
                score=round(float(r.score), 4),
                source=r.source or "unknown",
                content_snippet=r.content[:200] if r.content else "",
                metadata=meta,
            )
        )
    return docs


class EvaluationRunner:
    """Automated Evaluation Runner for 2 Retrievers x 2 LLMs."""

    def __init__(
        self,
        retrievers: dict[str, BaseRetriever] | None = None,
        llm_clients: dict[str, BaseLLMClient] | None = None,
        judge: LLMJudge | None = None,
        output_dir: str | Path = "data/evaluation/results",
        top_k: int = 5,
    ):
        self.retrievers = retrievers or {
            "vector": VectorRetriever(),
            "hybrid": HybridRetriever(),
        }
        self.llm_clients = llm_clients or {
            "model_1": get_eval_llm("model_1"),
            "model_2": get_eval_llm("model_2"),
        }
        self.judge = judge or LLMJudge(judge_model_name="model_1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k

    def run_single_combination(
        self,
        retrieval_key: str,
        llm_key: str,
        questions: list[EvalQuestion],
        on_progress: Any | None = None,
    ) -> list[SampleResult]:
        """Run evaluation for one (retriever, llm) combination across questions."""
        retriever = self.retrievers[retrieval_key]
        llm = self.llm_clients[llm_key]
        results: list[SampleResult] = []

        logger.info(f"==> Evaluating [{retrieval_key.upper()} Retrieval + {llm_key} ({llm.model_name})] ({len(questions)} questions)")

        for idx, q in enumerate(questions, 1):
            start_time = time.time()
            error_msg = None
            generated_answer = ""
            retrieved_docs_models: list[RetrievedDoc] = []
            context_str = ""

            logger.info(f"  [{idx}/{len(questions)}] Q{q.id}: \"{q.question[:50]}...\"")
            if on_progress:
                on_progress({
                    "type": "step",
                    "step": "retrieval",
                    "retrieval": retrieval_key,
                    "llm": llm_key,
                    "llm_name": llm.model_name,
                    "question_index": idx,
                    "total_questions": len(questions),
                    "question": q.question,
                    "message": f"[{retrieval_key} + {llm.model_name}] Q{idx}: Searching documents...",
                })

            try:
                # 1. Retrieve top-k documents
                search_results = retriever.search(q.question, top_k=self.top_k)
                retrieved_docs_models = convert_search_results(search_results)
                context_str = format_context(search_results)

                if on_progress:
                    on_progress({
                        "type": "step",
                        "step": "generation",
                        "retrieval": retrieval_key,
                        "llm": llm_key,
                        "llm_name": llm.model_name,
                        "question_index": idx,
                        "total_questions": len(questions),
                        "question": q.question,
                        "message": f"[{retrieval_key} + {llm.model_name}] Q{idx}: Generating response...",
                    })

                # 2. Generate answer with selected LLM
                generated_answer = llm.generate(query=q.question, context=context_str)

                if on_progress:
                    on_progress({
                        "type": "step",
                        "step": "evaluation",
                        "retrieval": retrieval_key,
                        "llm": llm_key,
                        "llm_name": llm.model_name,
                        "question_index": idx,
                        "total_questions": len(questions),
                        "question": q.question,
                        "message": f"[{retrieval_key} + {llm.model_name}] Q{idx}: LLM-as-Judge evaluating...",
                    })

            except Exception as e:
                logger.error(f"Error executing generation for Q{q.id}: {e}")
                error_msg = str(e)
                generated_answer = f"Error: {str(e)}"

            elapsed_ms = (time.time() - start_time) * 1000

            # 3. LLM Judge scoring
            judge_scores = self.judge.evaluate_sample(
                question=q.question,
                ground_truth=q.ground_truth,
                retrieved_context=context_str,
                generated_answer=generated_answer,
            )

            # 4. Retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(
                retrieved_docs=retrieved_docs_models,
                relevant_ids=q.relevant_documents,
                k=self.top_k,
            )

            sample_res = SampleResult(
                question_id=q.id,
                retrieval=retrieval_key,
                llm=llm_key,
                question=q.question,
                ground_truth=q.ground_truth,
                generated_answer=generated_answer,
                retrieved_documents=retrieved_docs_models,
                judge_scores=judge_scores,
                retrieval_metrics=retrieval_metrics,
                execution_time_ms=round(elapsed_ms, 2),
                error=error_msg,
            )
            results.append(sample_res)
            logger.info(
                f"     └─ Score: {judge_scores.overall}% (C:{judge_scores.correctness:.0f}%, F:{judge_scores.faithfulness:.0f}%, R:{judge_scores.relevance:.0f}%) | Time: {elapsed_ms:.0f}ms"
            )

            if on_progress:
                on_progress({
                    "type": "sample_complete",
                    "retrieval": retrieval_key,
                    "llm": llm_key,
                    "llm_name": llm.model_name,
                    "question_index": idx,
                    "total_questions": len(questions),
                    "question": q.question,
                    "score": judge_scores.overall,
                    "correctness": judge_scores.correctness,
                    "faithfulness": judge_scores.faithfulness,
                    "relevance": judge_scores.relevance,
                    "latency_ms": round(elapsed_ms, 1),
                    "message": f"✓ [{retrieval_key} + {llm.model_name}] Q{idx} Scored: {judge_scores.overall:.0f}% (Latency: {elapsed_ms:.0f}ms)",
                })

        return results

    def run_all_combinations(
        self,
        questions: list[EvalQuestion],
        retrieval_keys: list[str] | None = None,
        llm_keys: list[str] | None = None,
        dataset_path: str = "data/evaluation/evaluation_dataset.json",
        on_progress: Any | None = None,
    ) -> EvaluationReport:
        """Run full evaluation matrix across all selected combinations."""
        r_keys = retrieval_keys or list(self.retrievers.keys())
        l_keys = llm_keys or list(self.llm_clients.keys())

        all_detailed_results: list[SampleResult] = []
        summaries: dict[str, ConfigSummary] = {}
        matrix: dict[str, dict[str, float]] = {}

        total_combinations = len(r_keys) * len(l_keys)
        comb_idx = 0

        for r_key in r_keys:
            if r_key not in matrix:
                matrix[r_key] = {}
            for l_key in l_keys:
                comb_idx += 1
                comb_key = f"{r_key}+{l_key}"
                llm_name = self.llm_clients[l_key].model_name

                if on_progress:
                    on_progress({
                        "type": "combination_start",
                        "combination": comb_key,
                        "combination_index": comb_idx,
                        "total_combinations": total_combinations,
                        "retrieval": r_key,
                        "llm": l_key,
                        "llm_name": llm_name,
                        "message": f"▶ Starting Configuration [{comb_idx}/{total_combinations}]: {r_key.upper()} + {llm_name}",
                    })

                comb_results = self.run_single_combination(r_key, l_key, questions, on_progress=on_progress)
                all_detailed_results.extend(comb_results)

                # Compute aggregate summary for this combination
                n = len(comb_results)
                if n > 0:
                    avg_c = sum(r.judge_scores.correctness for r in comb_results) / n
                    avg_f = sum(r.judge_scores.faithfulness for r in comb_results) / n
                    avg_r = sum(r.judge_scores.relevance for r in comb_results) / n
                    avg_o = sum(r.judge_scores.overall for r in comb_results) / n
                    avg_lat = sum(r.execution_time_ms for r in comb_results) / n

                    annotated_prec = [r.retrieval_metrics.precision_at_k for r in comb_results if r.retrieval_metrics.precision_at_k is not None]
                    annotated_rec = [r.retrieval_metrics.recall_at_k for r in comb_results if r.retrieval_metrics.recall_at_k is not None]
                    annotated_mrr = [r.retrieval_metrics.mrr for r in comb_results if r.retrieval_metrics.mrr is not None]

                    avg_p = sum(annotated_prec) / len(annotated_prec) if annotated_prec else None
                    avg_rec = sum(annotated_rec) / len(annotated_rec) if annotated_rec else None
                    avg_m = sum(annotated_mrr) / len(annotated_mrr) if annotated_mrr else None
                else:
                    avg_c = avg_f = avg_r = avg_o = avg_lat = 0.0
                    avg_p = avg_rec = avg_m = None

                summary = ConfigSummary(
                    retrieval=r_key,
                    llm=l_key,
                    total_samples=n,
                    avg_correctness=round(avg_c, 2),
                    avg_faithfulness=round(avg_f, 2),
                    avg_relevance=round(avg_r, 2),
                    avg_overall=round(avg_o, 2),
                    avg_precision_at_k=round(avg_p, 4) if avg_p is not None else None,
                    avg_recall_at_k=round(avg_rec, 4) if avg_rec is not None else None,
                    avg_mrr=round(avg_m, 4) if avg_m is not None else None,
                    avg_latency_ms=round(avg_lat, 2),
                )
                summaries[comb_key] = summary
                matrix[r_key][l_key] = round(avg_o, 2)

        # Determine Best Configuration
        best_key = ""
        best_score = -1.0
        for k, s in summaries.items():
            if s.avg_overall > best_score:
                best_score = s.avg_overall
                best_key = k

        best_summary = summaries.get(best_key)
        best_reason = ""
        if best_summary:
            r_name = best_summary.retrieval.capitalize()
            l_name = best_summary.llm
            best_reason = (
                f"{r_name} Retrieval + {l_name} achieved the highest composite score ({best_summary.avg_overall}%) "
                f"with Correctness={best_summary.avg_correctness}%, Faithfulness={best_summary.avg_faithfulness}%, "
                f"Relevance={best_summary.avg_relevance}%, and average latency of {best_summary.avg_latency_ms:.1f}ms."
            )

        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_path=dataset_path,
            judge_model=self.judge.judge_model_name,
            top_k=self.top_k,
            configurations=summaries,
            best_configuration=best_key,
            best_reason=best_reason,
            comparison_matrix=matrix,
            detailed_results=all_detailed_results,
        )

        self.save_results(report)
        self.print_report(report)
        return report

    def save_results(self, report: EvaluationReport) -> None:
        """Save results to data/evaluation/results/."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Save main latest results
        latest_results_path = self.output_dir / "results.json"
        with open(latest_results_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

        # Save timestamped copy
        ts_results_path = self.output_dir / f"results_{timestamp_str}.json"
        with open(ts_results_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

        # Save concise summary
        summary_path = self.output_dir / "summary.json"
        summary_data = {
            "timestamp": report.timestamp,
            "dataset_path": report.dataset_path,
            "judge_model": report.judge_model,
            "best_configuration": report.best_configuration,
            "best_reason": report.best_reason,
            "comparison_matrix": report.comparison_matrix,
            "configurations": {k: v.model_dump() for k, v in report.configurations.items()},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation results to {latest_results_path} and {summary_path}")

    def print_report(self, report: EvaluationReport) -> None:
        """Print formatted CLI evaluation summary table and 2x2 matrix."""
        print("\n" + "=" * 80)
        print("           2 RETRIEVAL TECHNIQUES × 2 LLMS EVALUATION REPORT")
        print("=" * 80)
        print(f"Dataset:      {report.dataset_path}")
        print(f"Judge Model:  {report.judge_model}")
        print(f"Top-K:        {report.top_k}")
        print(f"Timestamp:    {report.timestamp}")
        print("-" * 80)

        # 2x2 Matrix Overview
        print("\n📊 2 × 2 OVERALL SCORE MATRIX (Composite Score 0-100%):")
        print("-" * 50)
        print(f"{'LLM':<18} | {'Vector Retrieval':<18} | {'Hybrid Retrieval':<18}")
        print("-" * 50)
        llm_keys = ["model_1", "model_2"]
        for l_k in llm_keys:
            vec_score = report.comparison_matrix.get("vector", {}).get(l_k, 0.0)
            hyb_score = report.comparison_matrix.get("hybrid", {}).get(l_k, 0.0)
            print(f"{l_k:<18} | {vec_score:>16.1f}% | {hyb_score:>16.1f}%")
        print("-" * 50)

        # Detailed metrics table
        print("\n📈 DETAILED CONFIGURATION COMPARISON:")
        print("-" * 105)
        header = f"{'Configuration':<22} | {'Correctness':<11} | {'Faithful':<9} | {'Relevance':<9} | {'Overall':<8} | {'P@K':<6} | {'R@K':<6} | {'MRR':<6} | {'Latency':<8}"
        print(header)
        print("-" * 105)
        for cfg_name, s in report.configurations.items():
            pk = f"{s.avg_precision_at_k:.2f}" if s.avg_precision_at_k is not None else "N/A"
            rk = f"{s.avg_recall_at_k:.2f}" if s.avg_recall_at_k is not None else "N/A"
            mrr = f"{s.avg_mrr:.2f}" if s.avg_mrr is not None else "N/A"
            row = (
                f"{cfg_name:<22} | "
                f"{s.avg_correctness:>10.1f}% | "
                f"{s.avg_faithfulness:>8.1f}% | "
                f"{s.avg_relevance:>8.1f}% | "
                f"{s.avg_overall:>7.1f}% | "
                f"{pk:>6} | "
                f"{rk:>6} | "
                f"{mrr:>6} | "
                f"{s.avg_latency_ms:>6.0f}ms"
            )
            print(row)
        print("-" * 105)

        # Best Configuration Callout
        print("\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   --> {report.best_configuration.upper()} <--")
        print(f"   Why: {report.best_reason}")
        print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="2 Retrievers x 2 LLMs RAG Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/evaluation_dataset.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K documents to retrieve (default: 5)",
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        choices=["vector", "hybrid", "both"],
        default="both",
        help="Retrieval strategy: vector, hybrid, or both",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["model_1", "model_2", "both"],
        default="both",
        help="LLM model: model_1, model_2, or both",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="model_1",
        help="LLM judge model (default: model_1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional limit on number of questions to evaluate",
    )

    args = parser.parse_args()

    questions = load_dataset(args.dataset)
    if args.max_questions:
        questions = questions[: args.max_questions]
    logger.info(f"Loaded {len(questions)} evaluation questions from {args.dataset}")

    retrieval_keys = ["vector", "hybrid"] if args.retrieval == "both" else [args.retrieval]
    llm_keys = ["model_1", "model_2"] if args.llm == "both" else [args.llm]

    runner = EvaluationRunner(
        output_dir=args.output,
        top_k=args.top_k,
        judge=LLMJudge(judge_model_name=args.judge),
    )

    runner.run_all_combinations(
        questions=questions,
        retrieval_keys=retrieval_keys,
        llm_keys=llm_keys,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()
