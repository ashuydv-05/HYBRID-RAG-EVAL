from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from src.evaluation.runner import EvaluationRunner, load_dataset

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

RESULTS_DIR = Path("data/evaluation/results")


import os
import queue
import threading
from fastapi.responses import StreamingResponse
from fastapi import Header

class RunEvaluationRequest(BaseModel):
    dataset_path: str = "data/evaluation/evaluation_dataset.json"
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval: str = Field(default="both", description="'vector', 'hybrid', or 'both'")
    llm: str = Field(default="both", description="'model_1', 'model_2', or 'both'")
    max_questions: int | None = Field(default=None, description="Optional question limit")
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None


@router.get("/health")
async def evaluation_health() -> dict[str, str]:
    """Health check for evaluation subsystem."""
    return {"status": "healthy", "service": "evaluation"}


@router.get("/config-info")
async def get_config_info() -> dict:
    """Return explicit model names and retrieval configurations for UI display."""
    return {
        "model_1": {
            "key": "model_1",
            "name": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            "provider": "Groq",
            "family": "Alibaba Qwen",
        },
        "model_2": {
            "key": "model_2",
            "name": os.getenv("GROQ_MODEL_2", "openai/gpt-oss-20b"),
            "provider": "Groq",
            "family": "OpenAI Family",
        },
        "judge": {
            "name": os.getenv("GEMINI_MODEL", "gemini-2.0-flash") if os.getenv("GEMINI_API_KEY") else os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            "provider": "Google Gemini" if os.getenv("GEMINI_API_KEY") else "Groq",
            "description": "LLM-as-Judge Evaluator",
        },
        "retrievers": {
            "vector": {
                "name": "Vector Retrieval (Dense)",
                "description": "Qdrant semantic vector search",
            },
            "hybrid": {
                "name": "Hybrid Retrieval (Dense + BM25 + RRF)",
                "description": "Reciprocal Rank Fusion across Qdrant & Elasticsearch BM25",
            },
        },
    }


@router.get("/summary")
async def get_evaluation_summary() -> dict:
    """Retrieve the latest aggregated evaluation summary and 2x2 matrix."""
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        return {
            "status": "not_found",
            "message": "No evaluation summary found. Please run an evaluation first.",
            "configurations": {},
            "comparison_matrix": {},
        }

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading evaluation summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation summary: {str(e)}")


@router.get("/results")
async def get_evaluation_results() -> dict:
    """Retrieve detailed per-question results from the latest evaluation."""
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        return {
            "status": "not_found",
            "message": "No evaluation results found. Please run an evaluation first.",
            "detailed_results": [],
        }

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading evaluation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {str(e)}")


@router.post("/stream")
async def stream_evaluation(
    request: RunEvaluationRequest,
    x_groq_api_key: str | None = Header(None, alias="x-groq-api-key"),
    x_gemini_api_key: str | None = Header(None, alias="x-gemini-api-key"),
    x_openai_api_key: str | None = Header(None, alias="x-openai-api-key"),
):
    """Execute evaluation and stream real-time progress events over SSE."""
    if x_groq_api_key or request.groq_api_key:
        os.environ["GROQ_API_KEY"] = x_groq_api_key or request.groq_api_key
    if x_gemini_api_key or request.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = x_gemini_api_key or request.gemini_api_key
    if x_openai_api_key or request.openai_api_key:
        os.environ["OPENAI_API_KEY"] = x_openai_api_key or request.openai_api_key

    def event_generator():
        event_queue = queue.Queue()

        def on_prog(data: dict):
            event_queue.put(data)

        def runner_target():
            try:
                questions = load_dataset(request.dataset_path)
                if request.max_questions:
                    questions = questions[: request.max_questions]

                retrieval_keys = ["vector", "hybrid"] if request.retrieval == "both" else [request.retrieval]
                llm_keys = ["model_1", "model_2"] if request.llm == "both" else [request.llm]

                runner = EvaluationRunner(output_dir=RESULTS_DIR, top_k=request.top_k)
                report = runner.run_all_combinations(
                    questions=questions,
                    retrieval_keys=retrieval_keys,
                    llm_keys=llm_keys,
                    dataset_path=request.dataset_path,
                    on_progress=on_prog,
                )
                event_queue.put({
                    "type": "complete",
                    "timestamp": report.timestamp,
                    "best_configuration": report.best_configuration,
                    "best_reason": report.best_reason,
                    "comparison_matrix": report.comparison_matrix,
                    "configurations": {k: v.model_dump() for k, v in report.configurations.items()},
                    "message": "✓ Evaluation Benchmark Completed Successfully!",
                })
            except Exception as exc:
                logger.error(f"Streaming evaluation error: {exc}", exc_info=True)
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=runner_target, daemon=True)
        thread.start()

        while True:
            item = event_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run")
async def run_evaluation(
    request: RunEvaluationRequest,
    x_groq_api_key: str | None = None,
    x_gemini_api_key: str | None = None,
    x_openai_api_key: str | None = None,
) -> dict:
    """Execute evaluation across selected configurations synchronously or with question limit."""
    try:
        if x_groq_api_key or request.groq_api_key:
            os.environ["GROQ_API_KEY"] = x_groq_api_key or request.groq_api_key
        if x_gemini_api_key or request.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = x_gemini_api_key or request.gemini_api_key
        if x_openai_api_key or request.openai_api_key:
            os.environ["OPENAI_API_KEY"] = x_openai_api_key or request.openai_api_key

        questions = load_dataset(request.dataset_path)
        if request.max_questions:
            questions = questions[: request.max_questions]

        retrieval_keys = ["vector", "hybrid"] if request.retrieval == "both" else [request.retrieval]
        llm_keys = ["model_1", "model_2"] if request.llm == "both" else [request.llm]

        runner = EvaluationRunner(output_dir=RESULTS_DIR, top_k=request.top_k)
        report = runner.run_all_combinations(
            questions=questions,
            retrieval_keys=retrieval_keys,
            llm_keys=llm_keys,
            dataset_path=request.dataset_path,
        )

        return {
            "status": "success",
            "timestamp": report.timestamp,
            "best_configuration": report.best_configuration,
            "best_reason": report.best_reason,
            "comparison_matrix": report.comparison_matrix,
            "configurations": {k: v.model_dump() for k, v in report.configurations.items()},
        }
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
