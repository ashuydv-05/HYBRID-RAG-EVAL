from __future__ import annotations

from typing import Sequence
from src.evaluation.models import RetrievalMetrics, RetrievedDoc


def compute_retrieval_metrics(
    retrieved_docs: Sequence[RetrievedDoc],
    relevant_ids: Sequence[str] | None,
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Precision@K, Recall@K, and MRR for a retrieved document list.

    If ground-truth relevant_ids are not provided or empty, reports that
    metrics are unavailable rather than inventing scores.
    """
    if not relevant_ids:
        return RetrievalMetrics(
            precision_at_k=None,
            recall_at_k=None,
            mrr=None,
            annotated=False,
            message="Retrieval metrics unavailable because ground-truth relevant documents are not annotated.",
        )

    # Normalize target relevant IDs (clean string, lower-case, strip)
    clean_target_ids = {str(pid).strip().lower() for pid in relevant_ids if str(pid).strip()}
    if not clean_target_ids:
        return RetrievalMetrics(
            precision_at_k=None,
            recall_at_k=None,
            mrr=None,
            annotated=False,
            message="Retrieval metrics unavailable because ground-truth relevant documents are not annotated.",
        )

    # Extract paper IDs from retrieved docs in order up to k
    retrieved_pids: list[str] = []
    for doc in retrieved_docs[:k]:
        pid = doc.paper_id or (doc.metadata.get("paper_id") if doc.metadata else None)
        if pid:
            retrieved_pids.append(str(pid).strip().lower())
        else:
            retrieved_pids.append("")

    # Precision@K
    relevant_retrieved_in_k = [pid for pid in retrieved_pids if pid in clean_target_ids]
    # To handle multiple chunks from same relevant paper, count unique relevant papers found
    unique_relevant_found = set(relevant_retrieved_in_k)
    p_at_k = len(relevant_retrieved_in_k) / float(k) if k > 0 else 0.0

    # Recall@K: proportion of all ground truth papers found in top k
    r_at_k = len(unique_relevant_found) / float(len(clean_target_ids)) if len(clean_target_ids) > 0 else 0.0

    # MRR: Reciprocal rank of the first relevant document hit
    mrr_score = 0.0
    for rank, pid in enumerate(retrieved_pids, start=1):
        if pid in clean_target_ids:
            mrr_score = 1.0 / float(rank)
            break

    return RetrievalMetrics(
        precision_at_k=round(p_at_k, 4),
        recall_at_k=round(r_at_k, 4),
        mrr=round(mrr_score, 4),
        annotated=True,
        message=None,
    )
