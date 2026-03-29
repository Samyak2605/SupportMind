from __future__ import annotations

import logging
from statistics import mean

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevance, context_precision, context_recall, faithfulness

from supportmind.config import load_config, load_settings
from supportmind.evaluation.dataset import create_evaluation_dataset
from supportmind.generation import SupportMindService
from supportmind.models import QueryRequest

logger = logging.getLogger(__name__)


def _run_variant(service: SupportMindService, dataset_rows: list[dict], mode: str, use_reranker: bool) -> dict:
    outputs = []
    latencies = []
    for row in dataset_rows:
        response = service.answer(QueryRequest(query=row["question"], mode=mode, use_reranker=use_reranker))
        outputs.append(
            {
                "question": row["question"],
                "answer": response.answer,
                "contexts": [chunk.text for chunk in response.retrieved_chunks],
                "ground_truth": row["ground_truth"],
            }
        )
        latencies.append(response.latency.total_ms)

    ragas_dataset = Dataset.from_pandas(pd.DataFrame(outputs))
    metrics = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevance, context_precision, context_recall],
    )
    result = metrics.to_pandas().mean(numeric_only=True).to_dict()
    result["average_latency_ms"] = mean(latencies)
    result["mode"] = mode
    result["reranker"] = use_reranker
    return result


def run_evaluation() -> pd.DataFrame:
    config = load_config()
    settings = load_settings()
    service = SupportMindService(config, settings)
    eval_rows = [row.model_dump() for row in create_evaluation_dataset(config)]

    variants = [
        ("semantic", False),
        ("hybrid", False),
        ("hybrid", True),
    ]

    results = [_run_variant(service, eval_rows, mode=mode, use_reranker=use_reranker) for mode, use_reranker in variants]
    frame = pd.DataFrame(results)
    logger.info("Evaluation completed")
    return frame
