from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_support_dataset(csv_path: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.dropna(subset=["instruction", "response", "category", "intent"]).reset_index(drop=True)
    logger.info("Dropped %s null rows", before - len(df))
    return df
