from __future__ import annotations

import logging
from itertools import islice

from supportmind.config import SupportMindConfig
from supportmind.ingestion.chunker import chunk_documents
from supportmind.ingestion.loader import load_support_dataset
from supportmind.ingestion.vectorstore import ChromaIndex, EmbeddingService
from supportmind.utils.io import write_jsonl

logger = logging.getLogger(__name__)


def _batched(items: list, batch_size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def build_vector_index(config: SupportMindConfig) -> int:
    df = load_support_dataset(str(config.paths.raw_data))
    rows = []
    for idx, record in df.iterrows():
        rows.append(
            {
                "row_id": int(idx),
                "instruction": record["instruction"],
                "response": record["response"],
                "category": record["category"],
                "intent": record["intent"],
            }
        )

    chunks = chunk_documents(
        rows=rows,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    logger.info("Generated %s chunks from %s records", len(chunks), len(rows))

    embedding_service = EmbeddingService(config.models.embedding_model)
    vector_index = ChromaIndex(str(config.paths.vectorstore_dir))
    vector_index.reset()

    batch_size = 128
    for batch_number, chunk_batch in enumerate(_batched(chunks, batch_size), start=1):
        embeddings = embedding_service.embed_documents([chunk.text for chunk in chunk_batch], batch_size=32)
        vector_index.upsert_chunks(chunk_batch, embeddings=embeddings, batch_size=batch_size)
        if batch_number % 25 == 0:
            logger.info("Indexed %s chunks", min(batch_number * batch_size, len(chunks)))

    write_jsonl(config.paths.processed_chunks, [chunk.model_dump() for chunk in chunks])
    logger.info("Vector index and processed chunk manifest saved successfully")
    return len(chunks)
