from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class Dataset:
    name: str
    description: str
    samples: List[dict[str, Any]] = field(default_factory=list)


class BenchmarkQueries:
    @staticmethod
    def get_general_queries() -> list[dict[str, Any]]:
        return [
            {"query": "What is the transformer architecture?", "expected_topics": ["attention", "transformer"]},
            {"query": "How does self-attention work?", "expected_topics": ["self-attention", "query", "key", "value"]},
            {"query": "What is BERT?", "expected_topics": ["BERT", "bidirectional", "transformers"]},
            {"query": "What is GPT?", "expected_topics": ["GPT", "generative", "autoregressive"]},
            {"query": "What is attention mechanism?", "expected_topics": ["attention", "neural networks"]},
            {"query": "What is layer normalization?", "expected_topics": ["normalization", "layer"]},
            {"query": "What is masked language modeling?", "expected_topics": ["masking", "language model"]},
            {"query": "What is encoder decoder?", "expected_topics": ["encoder", "decoder"]},
            {"query": "What is positional encoding?", "expected_topics": ["positional", "encoding"]},
            {"query": "What is scaled dot-product attention?", "expected_topics": ["scaled", "dot-product", "softmax"]},
        ]

    @staticmethod
    def get_advanced_queries() -> list[dict[str, Any]]:
        return [
            {"query": "Explain RoBERTa improvements over BERT.", "expected_topics": ["RoBERTa", "training"]},
            {"query": "How does multi-query attention differ from multi-head attention?", "expected_topics": ["multi-query", "multi-head"]},
            {"query": "Explain flash attention memory optimization.", "expected_topics": ["flash attention", "GPU", "memory"]},
            {"query": "How does LoRA fine-tuning work?", "expected_topics": ["LoRA", "low-rank", "adaptation"]},
            {"query": "What is direct preference optimization (DPO)?", "expected_topics": ["DPO", "preference", "RLHF"]},
        ]


class DatasetManager:
    def __init__(self, data_dir: str = "/tmp/test_evaluation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, name: str, description: str, samples: list[dict[str, Any]]) -> Dataset:
        return Dataset(name=name, description=description, samples=samples)

    def save_dataset(self, dataset: Dataset) -> str:
        filepath = self.data_dir / f"{dataset.name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"name": dataset.name, "description": dataset.description, "samples": dataset.samples}, f, indent=2)
        return str(filepath)

    def load_dataset(self, name: str) -> Dataset | None:
        filepath = self.data_dir / f"{name}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return Dataset(name=data["name"], description=data.get("description", ""), samples=data.get("samples", []))
