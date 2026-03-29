from __future__ import annotations

import logging

from supportmind.config import SupportMindConfig
from supportmind.ingestion.loader import load_support_dataset
from supportmind.models import EvaluationRow
from supportmind.utils.io import write_jsonl

logger = logging.getLogger(__name__)


def create_evaluation_dataset(config: SupportMindConfig) -> list[EvaluationRow]:
    df = load_support_dataset(str(config.paths.raw_data))
    sample = df.sample(n=config.evaluation.sample_size, random_state=config.evaluation.random_seed).reset_index(drop=True)
    rows = [
        EvaluationRow(
            question=row["instruction"],
            ground_truth=row["response"],
            category=row["category"],
            intent=row["intent"],
        )
        for _, row in sample.iterrows()
    ]
    write_jsonl(config.paths.evaluation_dataset, [row.model_dump() for row in rows])
    logger.info("Created evaluation dataset with %s rows", len(rows))
    return rows
